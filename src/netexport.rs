//! `--export-listen`: the export over HTTP, for monitoring a headless daemon
//! from another machine. Three read-only routes, hand-written HTTP/1.x:
//! `/stats` (latest snapshot), `/stream` (NDJSON, one line per export) and
//! `/health`. Everything runs on its own tasks; `publish` never waits.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, watch};
use tokio::time::timeout;

const HEAD_MAX: usize = 8 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONNS: usize = 16;
/// Exports a `/stream` client may fall behind by before it is dropped.
const LAG_MAX: usize = 32;
const LOG_401_EVERY: Duration = Duration::from_secs(60);

#[derive(Debug, PartialEq)]
pub struct Request {
    pub method: String,
    /// The path, without its query string.
    pub path: String,
    pub bearer: Option<String>,
}

/// Reads a request head (everything before the blank line). `Err` means a
/// malformed request: the reply is 400.
pub fn parse_request(head: &[u8]) -> Result<Request, ()> {
    let text = std::str::from_utf8(head).map_err(|_| ())?;
    let mut lines = text.split("\r\n");
    let mut first = lines.next().ok_or(())?.split(' ');
    let (method, target, version) = (first.next(), first.next(), first.next());
    let (Some(method), Some(target), Some(version), None) = (method, target, version, first.next())
    else {
        return Err(());
    };
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(());
    }
    if method.is_empty() || !target.starts_with('/') {
        return Err(());
    }
    let mut bearer = None;
    for line in lines.take_while(|l| !l.is_empty()) {
        let (name, value) = line.split_once(':').ok_or(())?;
        if name.eq_ignore_ascii_case("authorization") {
            let v = value.trim();
            if v.len() > 7 && v[..7].eq_ignore_ascii_case("bearer ") && v.is_char_boundary(7) {
                bearer = Some(v[7..].trim().to_string());
            }
        }
    }
    Ok(Request {
        method: method.to_string(),
        path: target.split('?').next().unwrap_or("/").to_string(),
        bearer,
    })
}

#[derive(Debug, PartialEq)]
pub enum Reply {
    Health,
    Stats,
    Stream,
    /// An error status: 401, 404 or 405.
    Status(u16),
}

/// Equal-time comparison, so a wrong token does not say how much matched.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= (a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)) as usize;
    }
    diff == 0
}

pub fn route(req: &Request, token: Option<&str>) -> Reply {
    if req.method != "GET" {
        return Reply::Status(405);
    }
    if req.path == "/health" {
        return Reply::Health;
    }
    if let Some(t) = token {
        match &req.bearer {
            Some(b) if ct_eq(b.as_bytes(), t.as_bytes()) => {}
            _ => return Reply::Status(401),
        }
    }
    match req.path.as_str() {
        "/stats" => Reply::Stats,
        "/stream" => Reply::Stream,
        _ => Reply::Status(404),
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        431 => "Request Header Fields Too Large",
        _ => "Service Unavailable",
    }
}

struct Ctx {
    token: Option<String>,
    latest: watch::Receiver<Option<Arc<str>>>,
    bcast: broadcast::Sender<Arc<str>>,
    events: mpsc::Sender<String>,
    shutdown: watch::Receiver<bool>,
    conns: AtomicUsize,
    streams: AtomicUsize,
    last_401: Mutex<HashMap<IpAddr, Instant>>,
}

impl Ctx {
    fn event(&self, msg: String) {
        // Full means nobody is reading the log (the dashboard is on screen).
        let _ = self.events.try_send(msg);
    }

    /// The head of a response; the body, if any, follows it.
    fn head(&self, status: u16, ctype: &str, len: Option<usize>) -> String {
        let mut h = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {ctype}\r\n",
            reason(status)
        );
        match len {
            Some(n) => h.push_str(&format!("Content-Length: {n}\r\n")),
            None => h.push_str("Transfer-Encoding: chunked\r\n"),
        }
        h.push_str("Cache-Control: no-store\r\nConnection: close\r\n");
        if self.token.is_none() {
            h.push_str("Access-Control-Allow-Origin: *\r\n");
        }
        match status {
            401 => h.push_str("WWW-Authenticate: Bearer\r\n"),
            405 => h.push_str("Allow: GET\r\n"),
            _ => {}
        }
        h.push_str("\r\n");
        h
    }

    fn response(&self, status: u16, ctype: &str, body: &str) -> Vec<u8> {
        let mut r = self.head(status, ctype, Some(body.len())).into_bytes();
        r.extend_from_slice(body.as_bytes());
        r
    }

    fn error(&self, status: u16) -> Vec<u8> {
        self.response(
            status,
            "text/plain",
            &format!("{status} {}\n", reason(status)),
        )
    }
}

async fn write(s: &mut TcpStream, data: &[u8]) -> bool {
    matches!(timeout(IO_TIMEOUT, s.write_all(data)).await, Ok(Ok(())))
}

enum HeadErr {
    TooLarge,
    Closed,
}

async fn read_head(s: &mut TcpStream) -> Result<Vec<u8>, HeadErr> {
    let mut buf = Vec::with_capacity(512);
    let mut chunk = [0u8; 1024];
    loop {
        let n = s.read(&mut chunk).await.map_err(|_| HeadErr::Closed)?;
        if n == 0 {
            return Err(HeadErr::Closed);
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            buf.truncate(end);
            return Ok(buf);
        }
        if buf.len() > HEAD_MAX {
            return Err(HeadErr::TooLarge);
        }
    }
}

struct ConnGuard(Arc<Ctx>);
impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.conns.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn handle(mut s: TcpStream, peer: SocketAddr, ctx: Arc<Ctx>) {
    let _guard = ConnGuard(Arc::clone(&ctx));
    let _ = s.set_nodelay(true);
    let head = match timeout(IO_TIMEOUT, read_head(&mut s)).await {
        Err(_) => {
            write(&mut s, &ctx.error(408)).await;
            return;
        }
        Ok(Err(HeadErr::TooLarge)) => {
            write(&mut s, &ctx.error(431)).await;
            return;
        }
        Ok(Err(HeadErr::Closed)) => return,
        Ok(Ok(h)) => h,
    };
    let Ok(req) = parse_request(&head) else {
        write(&mut s, &ctx.error(400)).await;
        return;
    };
    match route(&req, ctx.token.as_deref()) {
        Reply::Health => {
            write(&mut s, &ctx.response(200, "text/plain", "ok\n")).await;
        }
        Reply::Stats => {
            let latest = ctx.latest.borrow().clone();
            let out = match latest {
                Some(l) => ctx.response(200, "application/json", &l),
                None => ctx.error(503),
            };
            write(&mut s, &out).await;
        }
        Reply::Status(status) => {
            if status == 401 {
                let due = {
                    let mut seen = ctx.last_401.lock().unwrap_or_else(|e| e.into_inner());
                    let now = Instant::now();
                    seen.retain(|_, t| now.duration_since(*t) < LOG_401_EVERY);
                    seen.insert(peer.ip(), now).is_none()
                };
                if due {
                    ctx.event(format!("401 from {peer}"));
                }
            }
            write(&mut s, &ctx.error(status)).await;
        }
        Reply::Stream => stream(s, peer, &ctx).await,
    }
}

async fn chunk(s: &mut TcpStream, line: &str) -> bool {
    let mut out = format!("{:X}\r\n", line.len()).into_bytes();
    out.extend_from_slice(line.as_bytes());
    out.extend_from_slice(b"\r\n");
    write(s, &out).await
}

async fn stream(mut s: TcpStream, peer: SocketAddr, ctx: &Arc<Ctx>) {
    // Subscribed before the latest line is read, so no export falls between.
    let mut rx = ctx.bcast.subscribe();
    let mut stop = ctx.shutdown.clone();
    let n = ctx.streams.fetch_add(1, Ordering::SeqCst) + 1;
    ctx.event(format!("client {peer} connected to /stream ({n} now)"));
    let head = ctx.head(200, "application/x-ndjson", None);
    let mut alive = write(&mut s, head.as_bytes()).await;
    let mut last: Option<Arc<str>> = None;
    if alive {
        let first = ctx.latest.borrow().clone();
        if let Some(line) = first {
            alive = chunk(&mut s, &line).await;
            last = Some(line);
        }
    }
    let mut clean = false;
    while alive {
        if *stop.borrow() {
            clean = true;
            break;
        }
        tokio::select! {
            r = rx.recv() => match r {
                Ok(line) => {
                    // The same export the latest line already sent.
                    if last.as_ref().is_some_and(|l| Arc::ptr_eq(l, &line)) {
                        continue;
                    }
                    alive = chunk(&mut s, &line).await;
                    last = Some(line);
                }
                // Too far behind: dropped, not waited for.
                Err(broadcast::error::RecvError::Lagged(_)) => break,
                Err(broadcast::error::RecvError::Closed) => {
                    clean = true;
                    break;
                }
            },
            _ = stop.changed() => {
                clean = true;
                break;
            }
        }
    }
    if clean {
        write(&mut s, b"0\r\n\r\n").await;
    }
    let n = ctx.streams.fetch_sub(1, Ordering::SeqCst) - 1;
    ctx.event(format!("client {peer} disconnected from /stream ({n} now)"));
}

async fn accept_loop(listener: TcpListener, ctx: Arc<Ctx>) {
    let mut stop = ctx.shutdown.clone();
    loop {
        tokio::select! {
            r = listener.accept() => match r {
                Ok((s, peer)) => {
                    if ctx.conns.fetch_add(1, Ordering::SeqCst) >= MAX_CONNS {
                        ctx.conns.fetch_sub(1, Ordering::SeqCst);
                        let ctx = Arc::clone(&ctx);
                        tokio::spawn(async move {
                            let mut s = s;
                            write(&mut s, &ctx.error(503)).await;
                        });
                    } else {
                        tokio::spawn(handle(s, peer, Arc::clone(&ctx)));
                    }
                }
                Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
            },
            _ = stop.changed() => break,
        }
    }
    // The listener drops here: the port is free once this returns.
}

pub struct HttpServer {
    addr: SocketAddr,
    auth: bool,
    latest: watch::Sender<Option<Arc<str>>>,
    bcast: broadcast::Sender<Arc<str>>,
    events: mpsc::Receiver<String>,
    stop: watch::Sender<bool>,
    ctx: Arc<Ctx>,
    accept: Option<tokio::task::JoinHandle<()>>,
}

impl HttpServer {
    /// Binds now, so a bad address is a startup error; serves from a task.
    /// Call from inside the tokio runtime.
    pub fn bind(listen: &str, token: Option<String>) -> Result<Self, String> {
        let err = |e: std::io::Error| format!("{listen}: {e}");
        let std_listener = std::net::TcpListener::bind(listen).map_err(err)?;
        std_listener.set_nonblocking(true).map_err(err)?;
        let addr = std_listener.local_addr().map_err(err)?;
        let listener = TcpListener::from_std(std_listener).map_err(err)?;
        let (latest, latest_rx) = watch::channel(None);
        let (bcast, _) = broadcast::channel(LAG_MAX);
        let (events_tx, events) = mpsc::channel(256);
        let (stop, stop_rx) = watch::channel(false);
        let auth = token.is_some();
        let ctx = Arc::new(Ctx {
            token,
            latest: latest_rx,
            bcast: bcast.clone(),
            events: events_tx,
            shutdown: stop_rx,
            conns: AtomicUsize::new(0),
            streams: AtomicUsize::new(0),
            last_401: Mutex::new(HashMap::new()),
        });
        let accept = tokio::spawn(accept_loop(listener, Arc::clone(&ctx)));
        Ok(Self {
            addr,
            auth,
            latest,
            bcast,
            events,
            stop,
            ctx,
            accept: Some(accept),
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn auth(&self) -> bool {
        self.auth
    }

    /// Hands one export to `/stats` and to every `/stream` client.
    pub fn publish(&self, line: Arc<str>) {
        let _ = self.bcast.send(Arc::clone(&line));
        self.latest.send_replace(Some(line));
    }

    /// Connection events since the last call, for the log.
    pub fn take_events(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(e) = self.events.try_recv() {
            out.push(e);
        }
        out
    }

    /// Stops accepting, ends the streams with their closing chunk and waits
    /// (a second at most) for the connections to finish.
    pub async fn close(&mut self) {
        let _ = self.stop.send(true);
        if let Some(h) = self.accept.take() {
            let _ = timeout(Duration::from_secs(1), h).await;
        }
        let end = Instant::now() + Duration::from_secs(1);
        while self.ctx.conns.load(Ordering::SeqCst) > 0 && Instant::now() < end {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::Exporter;
    use crate::perf::PerfTracker;
    use tokio::io::{AsyncBufReadExt, BufReader};

    fn req(text: &str) -> Result<Request, ()> {
        parse_request(text.as_bytes())
    }

    #[test]
    fn parses_method_path_query_and_bearer() {
        let r = req("GET /stats?x=1 HTTP/1.1\r\nHost: h\r\nauthorization: BEARER  abc \r\nX: y")
            .unwrap();
        assert_eq!(r.method, "GET");
        assert_eq!(r.path, "/stats");
        assert_eq!(r.bearer.as_deref(), Some("abc"));
        assert!(req("GET /stats HTTP/1.0").unwrap().bearer.is_none());
        // A scheme other than Bearer is no token.
        let r = req("GET / HTTP/1.1\r\nAuthorization: Basic abc").unwrap();
        assert!(r.bearer.is_none());
    }

    #[test]
    fn rejects_malformed_requests() {
        assert!(req("").is_err());
        assert!(req("GET /stats").is_err());
        assert!(req("GET /stats HTTP/2").is_err());
        assert!(req("GET stats HTTP/1.1").is_err());
        assert!(req("GET /a /b HTTP/1.1").is_err());
        assert!(req("GET / HTTP/1.1\r\nno colon here").is_err());
        assert!(parse_request(b"GET / HTTP/1.1\r\nX: \xff\xfe").is_err());
    }

    fn r(method: &str, path: &str, bearer: Option<&str>) -> Request {
        Request {
            method: method.into(),
            path: path.into(),
            bearer: bearer.map(String::from),
        }
    }

    #[test]
    fn routes_and_auth() {
        let t = Some("secret");
        assert_eq!(route(&r("GET", "/health", None), t), Reply::Health);
        assert_eq!(route(&r("GET", "/stats", None), None), Reply::Stats);
        assert_eq!(route(&r("GET", "/stream", None), None), Reply::Stream);
        assert_eq!(route(&r("GET", "/stats", None), t), Reply::Status(401));
        assert_eq!(
            route(&r("GET", "/stats", Some("nope")), t),
            Reply::Status(401)
        );
        assert_eq!(
            route(&r("GET", "/stats", Some("secret1")), t),
            Reply::Status(401)
        );
        assert_eq!(route(&r("GET", "/stats", Some("secret")), t), Reply::Stats);
        assert_eq!(
            route(&r("GET", "/stream", Some("secret")), t),
            Reply::Stream
        );
        assert_eq!(route(&r("GET", "/other", None), None), Reply::Status(404));
        // An unknown path does not tell an unauthenticated caller it is unknown.
        assert_eq!(route(&r("GET", "/other", None), t), Reply::Status(401));
        assert_eq!(route(&r("POST", "/stats", None), None), Reply::Status(405));
        assert_eq!(route(&r("HEAD", "/health", None), None), Reply::Status(405));
    }

    #[test]
    fn constant_time_compare() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
        assert!(!ct_eq(b"a", b"a\0"));
        assert!(ct_eq(b"", b""));
    }

    /// One request, read to the end of the response.
    async fn get(addr: SocketAddr, path: &str, token: Option<&str>) -> (String, String) {
        let mut s = TcpStream::connect(addr).await.unwrap();
        let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
        s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n{auth}\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = Vec::new();
        timeout(Duration::from_secs(5), s.read_to_end(&mut out))
            .await
            .unwrap()
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        (head.to_string(), body.to_string())
    }

    /// The next chunk of a chunked body; `None` at the closing chunk or EOF.
    async fn next_line(r: &mut BufReader<TcpStream>) -> Option<String> {
        let mut size = String::new();
        timeout(Duration::from_secs(5), r.read_line(&mut size))
            .await
            .ok()?
            .ok()?;
        let n = usize::from_str_radix(size.trim(), 16).ok()?;
        if n == 0 {
            return None;
        }
        let mut data = vec![0u8; n + 2];
        timeout(Duration::from_secs(5), r.read_exact(&mut data))
            .await
            .ok()?
            .ok()?;
        Some(String::from_utf8(data).unwrap())
    }

    async fn open_stream(addr: SocketAddr, token: &str) -> BufReader<TcpStream> {
        let mut s = TcpStream::connect(addr).await.unwrap();
        let q = format!("GET /stream HTTP/1.1\r\nAuthorization: Bearer {token}\r\n\r\n");
        s.write_all(q.as_bytes()).await.unwrap();
        let mut r = BufReader::new(s);
        let mut head = String::new();
        loop {
            let mut l = String::new();
            r.read_line(&mut l).await.unwrap();
            if l == "\r\n" {
                break;
            }
            head.push_str(&l);
        }
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        assert!(head.contains("application/x-ndjson"));
        assert!(head.contains("Transfer-Encoding: chunked"));
        r
    }

    fn tick(ex: &mut Exporter) {
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let perf = PerfTracker::new();
        let rows = std::iter::once((&model, None, &perf, 0, 8192));
        ex.tick(rows, &[], Instant::now());
    }

    fn server() -> (Exporter, SocketAddr) {
        let ex = Exporter::open(None, None, Duration::ZERO, "live")
            .unwrap()
            .with_http("127.0.0.1:0", Some("tok".into()))
            .unwrap();
        let addr = ex.listen_info().unwrap().0;
        (ex, addr)
    }

    #[tokio::test]
    async fn health_stats_and_auth() {
        let (mut ex, addr) = server();
        let (head, body) = get(addr, "/health", None).await;
        assert!(head.starts_with("HTTP/1.1 200"));
        assert_eq!(body, "ok\n");
        let (head, _) = get(addr, "/stats", None).await;
        assert!(head.starts_with("HTTP/1.1 401"));
        assert!(head.contains("WWW-Authenticate: Bearer"));
        let (head, _) = get(addr, "/stats", Some("wrong")).await;
        assert!(head.starts_with("HTTP/1.1 401"));
        let (head, _) = get(addr, "/stats", Some("tok")).await;
        assert!(
            head.starts_with("HTTP/1.1 503"),
            "no snapshot before the first tick"
        );
        tick(&mut ex);
        let (head, body) = get(addr, "/stats?pretty=1", Some("tok")).await;
        assert!(head.starts_with("HTTP/1.1 200"));
        assert!(head.contains("application/json"));
        assert!(head.contains("Cache-Control: no-store"));
        assert!(
            !head.contains("Access-Control-Allow-Origin"),
            "no CORS with a token"
        );
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["schema"], "llm-visuals.stats/1");
        let (head, _) = get(addr, "/nope", Some("tok")).await;
        assert!(head.starts_with("HTTP/1.1 404"));
        let ev = ex.take_events();
        assert_eq!(
            ev.iter().filter(|e| e.starts_with("401 from")).count(),
            1,
            "{ev:?}"
        );
        ex.close().await;
    }

    #[tokio::test]
    async fn cors_only_without_token() {
        let mut ex = Exporter::open(None, None, Duration::ZERO, "live")
            .unwrap()
            .with_http("127.0.0.1:0", None)
            .unwrap();
        tick(&mut ex);
        let addr = ex.listen_info().unwrap().0;
        let (head, _) = get(addr, "/stats", None).await;
        assert!(head.contains("Access-Control-Allow-Origin: *"));
        ex.close().await;
    }

    #[tokio::test]
    async fn stream_gets_latest_then_every_export_and_ends_cleanly() {
        let (mut ex, addr) = server();
        tick(&mut ex);
        let mut r = open_stream(addr, "tok").await;
        let first: serde_json::Value =
            serde_json::from_str(&next_line(&mut r).await.unwrap()).unwrap();
        tick(&mut ex);
        let second: serde_json::Value =
            serde_json::from_str(&next_line(&mut r).await.unwrap()).unwrap();
        tick(&mut ex);
        let third: serde_json::Value =
            serde_json::from_str(&next_line(&mut r).await.unwrap()).unwrap();
        assert_eq!(first["seq"], 0);
        assert_eq!(second["seq"], 1);
        assert_eq!(third["seq"], 2, "no line twice, none missed");
        assert!(ex
            .take_events()
            .iter()
            .any(|e| e.contains("connected to /stream (1 now)")));
        // A clean shutdown sends the closing chunk.
        ex.close().await;
        assert!(next_line(&mut r).await.is_none());
        let mut rest = Vec::new();
        r.read_to_end(&mut rest).await.unwrap();
        assert!(TcpStream::connect(addr).await.is_err(), "port is free");
    }

    #[tokio::test]
    async fn lagging_stream_client_is_dropped_without_blocking_tick() {
        let (mut ex, addr) = server();
        tick(&mut ex);
        let mut r = open_stream(addr, "tok").await;
        assert!(next_line(&mut r).await.is_some());
        // The client task cannot run while this loop does: it falls behind.
        let started = Instant::now();
        for _ in 0..(LAG_MAX * 4) {
            tick(&mut ex);
        }
        assert!(started.elapsed() < Duration::from_secs(1));
        tokio::time::sleep(Duration::from_millis(200)).await;
        // Dropped: what was already sent drains, then the connection ends
        // with no closing chunk.
        let mut lines = 0;
        while next_line(&mut r).await.is_some() {
            lines += 1;
            assert!(lines <= LAG_MAX * 4);
        }
        let ev = ex.take_events();
        assert!(
            ev.iter()
                .any(|e| e.contains("disconnected from /stream (0 now)")),
            "{ev:?}"
        );
        ex.close().await;
    }

    #[tokio::test]
    async fn oversized_head_and_bad_requests() {
        let (mut ex, addr) = server();
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(10_000)).as_bytes())
            .await
            .unwrap();
        let mut out = Vec::new();
        timeout(Duration::from_secs(5), s.read_to_end(&mut out))
            .await
            .unwrap()
            .ok();
        assert!(String::from_utf8_lossy(&out).starts_with("HTTP/1.1 431"));
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(b"GARBAGE\r\n\r\n").await.unwrap();
        let mut out = Vec::new();
        timeout(Duration::from_secs(5), s.read_to_end(&mut out))
            .await
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&out).starts_with("HTTP/1.1 400"));
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(b"POST /stats HTTP/1.1\r\n\r\n").await.unwrap();
        let mut out = Vec::new();
        timeout(Duration::from_secs(5), s.read_to_end(&mut out))
            .await
            .unwrap()
            .unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(text.starts_with("HTTP/1.1 405") && text.contains("Allow: GET"));
        ex.close().await;
    }

    #[tokio::test]
    async fn too_many_connections_get_503() {
        let (mut ex, addr) = server();
        // Sixteen silent connections fill the slots (each waits for a head).
        let mut held = Vec::new();
        for _ in 0..MAX_CONNS {
            held.push(TcpStream::connect(addr).await.unwrap());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut s = TcpStream::connect(addr).await.unwrap();
        let mut out = Vec::new();
        timeout(Duration::from_secs(5), s.read_to_end(&mut out))
            .await
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&out).starts_with("HTTP/1.1 503"));
        drop(held);
        ex.close().await;
    }

    #[tokio::test]
    async fn bind_error_is_reported() {
        let first = HttpServer::bind("127.0.0.1:0", None).unwrap();
        let taken = first.addr().to_string();
        assert!(HttpServer::bind(&taken, None).is_err());
    }
}
