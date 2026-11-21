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
    /// The `Host` header, as sent.
    pub host: Option<String>,
    pub http11: bool,
}

/// The host of `host[:port]`, `[v6][:port]` or a bare IPv6 address, without
/// brackets or port. `None` if the text is not shaped like one of those.
fn host_part(s: &str) -> Option<&str> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix('[') {
        let (inner, tail) = rest.split_once(']')?;
        let port_ok = tail.is_empty()
            || tail
                .strip_prefix(':')
                .is_some_and(|p| p.bytes().all(|b| b.is_ascii_digit()));
        return port_ok.then_some(inner);
    }
    if s.parse::<IpAddr>().is_ok() {
        return Some(s); // a bare IPv6 address
    }
    match s.rsplit_once(':') {
        Some((h, p)) if !h.contains(':') && p.bytes().all(|b| b.is_ascii_digit()) => Some(h),
        Some(_) => None,
        None => Some(s),
    }
}

/// The `Host` names a request may carry when there is no token. Exact
/// names only: a suffix or wildcard would let a name the attacker controls
/// (`gpubox1.evil.com`) through.
#[derive(Debug, Clone, Default)]
pub struct HostPolicy {
    /// The host part of `--export-listen` (empty for an address).
    listen_host: String,
    /// This machine's hostname as the system gives it, and its first DNS
    /// label, lower case.
    own: Vec<String>,
    /// `--export-allow-host`, lower case.
    extra: Vec<String>,
}

fn normalized(name: &str) -> String {
    name.trim().trim_end_matches('.').to_ascii_lowercase()
}

impl HostPolicy {
    pub fn new(listen_host: &str, hostname: Option<&str>, extra: &[String]) -> Self {
        let mut own = Vec::new();
        if let Some(full) = hostname.map(normalized).filter(|h| !h.is_empty()) {
            let label = full.split('.').next().unwrap_or("").to_string();
            own.push(full);
            if !label.is_empty() && !own.contains(&label) {
                own.push(label);
            }
        }
        Self {
            listen_host: normalized(listen_host),
            own,
            extra: extra.iter().map(|e| normalized(e)).collect(),
        }
    }

    /// For the startup log.
    pub fn describe(&self) -> String {
        let mut v = vec!["IP addresses".to_string(), "localhost".to_string()];
        if !self.listen_host.is_empty() && self.listen_host.parse::<IpAddr>().is_err() {
            v.push(self.listen_host.clone());
        }
        v.extend(self.own.iter().cloned());
        v.extend(self.extra.iter().cloned());
        v.dedup();
        v.join(", ")
    }

    /// Whether a request for `Host: header` may be answered when there is no
    /// token: an IP literal, `localhost`, the host `--export-listen` names,
    /// this machine's hostname or its first label, or a name given with
    /// `--export-allow-host`; all exact. A name that points at this address
    /// from someone else's page (DNS rebinding) is none of these.
    fn allows(&self, header: &str) -> bool {
        let Some(h) = host_part(header) else {
            return false;
        };
        if h.parse::<IpAddr>().is_ok() || h.eq_ignore_ascii_case("localhost") {
            return true;
        }
        let name = normalized(h);
        !name.is_empty()
            && (name == self.listen_host || self.own.contains(&name) || self.extra.contains(&name))
    }
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
    let mut host = None;
    for line in lines.take_while(|l| !l.is_empty()) {
        let (name, value) = line.split_once(':').ok_or(())?;
        if name.eq_ignore_ascii_case("authorization") {
            let v = value.trim();
            // Bytes, not str slicing: a multibyte character may straddle 7.
            if v.as_bytes()
                .get(..7)
                .is_some_and(|p| p.eq_ignore_ascii_case(b"bearer "))
            {
                bearer = Some(
                    String::from_utf8_lossy(&v.as_bytes()[7..])
                        .trim()
                        .to_string(),
                );
            }
        } else if name.eq_ignore_ascii_case("host") && host.is_none() {
            host = Some(value.trim().to_string());
        }
    }
    Ok(Request {
        method: method.to_string(),
        path: target.split('?').next().unwrap_or("/").to_string(),
        bearer,
        host,
        http11: version == "HTTP/1.1",
    })
}

#[derive(Debug, PartialEq)]
pub enum Reply {
    Health,
    Stats,
    Stream,
    /// An error status: 400, 401, 403, 404 or 405.
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

pub fn route(req: &Request, token: Option<&str>, hosts: &HostPolicy) -> Reply {
    if req.method != "GET" {
        return Reply::Status(405);
    }
    // With a token, the token is the protection. Without one, only a request
    // that names this machine by an address, `localhost` or the listen host.
    if token.is_none() {
        match &req.host {
            None if req.http11 => return Reply::Status(400),
            Some(h) if !hosts.allows(h) => return Reply::Status(403),
            _ => {}
        }
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
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        431 => "Request Header Fields Too Large",
        _ => "Service Unavailable",
    }
}

struct Ctx {
    token: Option<String>,
    /// The host part of `--export-listen`, for the `Host` check.
    hosts: HostPolicy,
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

    /// The head of a response; the body, if any, follows it. With no
    /// length the body is chunked, or for an HTTP/1.0 client (which has no
    /// chunked encoding) runs until the connection closes.
    fn head(&self, status: u16, ctype: &str, len: Option<usize>, http11: bool) -> String {
        let mut h = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {ctype}\r\n",
            reason(status)
        );
        match len {
            Some(n) => h.push_str(&format!("Content-Length: {n}\r\n")),
            None if http11 => h.push_str("Transfer-Encoding: chunked\r\n"),
            None => {}
        }
        h.push_str("Cache-Control: no-store\r\nConnection: close\r\n");
        match status {
            401 => h.push_str("WWW-Authenticate: Bearer\r\n"),
            405 => h.push_str("Allow: GET\r\n"),
            _ => {}
        }
        h.push_str("\r\n");
        h
    }

    fn response(&self, status: u16, ctype: &str, body: &str) -> Vec<u8> {
        let mut r = self
            .head(status, ctype, Some(body.len()), true)
            .into_bytes();
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

/// How long a closing connection may keep draining what the client still
/// sends, and how much of it we read.
const DRAIN_TIME: Duration = Duration::from_secs(1);
const DRAIN_MAX: usize = 64 * 1024;

/// Writes a final reply, then closes the connection gracefully: FIN first,
/// then discard what the client still sends. Dropping a socket that has unread
/// bytes sends a RST on Windows, which can destroy the reply before the
/// client reads it.
async fn write_and_close(s: &mut TcpStream, data: &[u8], drain: Duration) {
    if !write(s, data).await {
        return;
    }
    let _ = s.shutdown().await;
    let mut left = DRAIN_MAX;
    let mut buf = [0u8; 4096];
    let _ = timeout(drain, async {
        while left > 0 {
            match s.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => left = left.saturating_sub(n),
            }
        }
    })
    .await;
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
            write_and_close(&mut s, &ctx.error(408), DRAIN_TIME).await;
            return;
        }
        Ok(Err(HeadErr::TooLarge)) => {
            write_and_close(&mut s, &ctx.error(431), DRAIN_TIME).await;
            return;
        }
        Ok(Err(HeadErr::Closed)) => return,
        Ok(Ok(h)) => h,
    };
    let Ok(req) = parse_request(&head) else {
        write_and_close(&mut s, &ctx.error(400), DRAIN_TIME).await;
        return;
    };
    match route(&req, ctx.token.as_deref(), &ctx.hosts) {
        Reply::Health => {
            write_and_close(&mut s, &ctx.response(200, "text/plain", "ok\n"), DRAIN_TIME).await;
        }
        Reply::Stats => {
            let latest = ctx.latest.borrow().clone();
            let out = match latest {
                Some(l) => ctx.response(200, "application/json", &l),
                None => ctx.error(503),
            };
            write_and_close(&mut s, &out, DRAIN_TIME).await;
        }
        Reply::Status(status) => {
            if status == 401 {
                let due = {
                    let mut seen = ctx.last_401.lock().unwrap_or_else(|e| e.into_inner());
                    let now = Instant::now();
                    seen.retain(|_, t| now.duration_since(*t) < LOG_401_EVERY);
                    // Stamped only when logged, so a client that keeps
                    // trying is logged again once a period has passed.
                    let due = !seen.contains_key(&peer.ip());
                    if due {
                        seen.insert(peer.ip(), now);
                    }
                    due
                };
                if due {
                    ctx.event(format!("401 from {peer}"));
                }
            }
            write_and_close(&mut s, &ctx.error(status), DRAIN_TIME).await;
        }
        Reply::Stream => stream(s, peer, &ctx, req.http11).await,
    }
}

/// One NDJSON line: a chunk for HTTP/1.1, the bare line for HTTP/1.0.
async fn chunk(s: &mut TcpStream, line: &str, chunked: bool) -> bool {
    if !chunked {
        return write(s, line.as_bytes()).await;
    }
    let mut out = format!("{:X}\r\n", line.len()).into_bytes();
    out.extend_from_slice(line.as_bytes());
    out.extend_from_slice(b"\r\n");
    write(s, &out).await
}

async fn stream(mut s: TcpStream, peer: SocketAddr, ctx: &Arc<Ctx>, chunked: bool) {
    // Subscribed before the latest line is read, so no export falls between.
    let mut rx = ctx.bcast.subscribe();
    let mut stop = ctx.shutdown.clone();
    let n = ctx.streams.fetch_add(1, Ordering::SeqCst) + 1;
    ctx.event(format!("client {peer} connected to /stream ({n} now)"));
    let head = ctx.head(200, "application/x-ndjson", None, chunked);
    let mut alive = write(&mut s, head.as_bytes()).await;
    let mut last: Option<Arc<str>> = None;
    if alive {
        let first = ctx.latest.borrow().clone();
        if let Some(line) = first {
            alive = chunk(&mut s, &line, chunked).await;
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
                    alive = chunk(&mut s, &line, chunked).await;
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
    if clean && chunked {
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
                            write_and_close(&mut s, &ctx.error(503), Duration::from_millis(250)).await;
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
    pub fn bind(
        listen: &str,
        token: Option<String>,
        allow_hosts: &[String],
    ) -> Result<Self, String> {
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
        let hosts = HostPolicy::new(
            host_part(listen).unwrap_or_default(),
            crate::export::hostname().as_deref(),
            allow_hosts,
        );
        let ctx = Arc::new(Ctx {
            hosts,
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

    /// The `Host` names answered without a token; `None` with one.
    pub fn accepted_hosts(&self) -> Option<String> {
        self.ctx.token.is_none().then(|| self.ctx.hosts.describe())
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
            host: Some("localhost:9464".into()),
            http11: true,
        }
    }

    fn policy() -> HostPolicy {
        HostPolicy::new(
            "box.lan",
            Some("gpubox1.example.lan"),
            &["NAS.lan".to_string()],
        )
    }

    fn route(req: &Request, token: Option<&str>) -> Reply {
        super::route(req, token, &policy())
    }

    #[test]
    fn own_hostname_and_allowed_names() {
        // policy(): hostname "gpubox1.example.lan", --export-allow-host NAS.lan.
        for ok in [
            "gpubox1.example.lan",
            "Gpubox1.EXAMPLE.LAN:9464",
            "gpubox1.example.lan.",
            "gpubox1",
            "GPUBOX1:9464",
            "nas.lan",
            "NAS.LAN:80",
            "box.lan",
        ] {
            assert!(policy().allows(ok), "{ok}");
        }
        for bad in [
            // A suffix on the machine's name is a name someone else may own.
            "gpubox1.evil.com",
            "gpubox1.example.org",
            "gpubox1.example.lan.evil.com",
            "gpubox1.example.lan.example",
            "gpubox",
            "gpubox10",
            "gpubox1x.example.lan",
            "xgpubox1.example.lan",
            "evil.gpubox1.example.lan",
            "gpu-box1.example.lan",
            "nas",
            "nas.lan.evil.com",
            "x.nas.lan",
            ".example.lan",
            "..",
        ] {
            assert!(!policy().allows(bad), "{bad}");
        }
        // A name added with --export-allow-host is exact, not a suffix.
        let p = HostPolicy::new(
            "",
            Some("rack7-gpubox2.example.lan"),
            &["gpubox1.example.org".into()],
        );
        assert!(p.allows("rack7-gpubox2.example.lan") && p.allows("rack7-gpubox2"));
        assert!(p.allows("gpubox1.example.org"));
        assert!(!p.allows("gpubox1.example.org.evil.com") && !p.allows("gpubox1"));
        // A hostname without a domain is its own label.
        let p = HostPolicy::new("", Some("Box9\n"), &[]);
        assert!(p.allows("box9") && p.allows("BOX9:80") && !p.allows("box9.evil.com"));
        // No hostname known, none given: only literals and localhost.
        let p = HostPolicy::new("", None, &[]);
        assert!(p.allows("127.0.0.1") && p.allows("localhost") && !p.allows("gpubox1"));
        assert!(!HostPolicy::new("", Some(""), &[]).allows(""));
        assert_eq!(
            policy().describe(),
            "IP addresses, localhost, box.lan, gpubox1.example.lan, gpubox1, nas.lan"
        );
        // No Host check at all with a token.
        let mut q = r("GET", "/stats", Some("t"));
        q.host = Some("gpubox1.evil.com".into());
        assert_eq!(route(&q, Some("t")), Reply::Stats);
        q.bearer = None;
        assert_eq!(route(&q, None), Reply::Status(403));
        q.host = Some("gpubox1.example.lan:9464".into());
        assert_eq!(route(&q, None), Reply::Stats);
    }

    #[test]
    fn host_header_rules_without_a_token() {
        for ok in [
            "127.0.0.1",
            "127.0.0.1:9464",
            "10.1.2.3:80",
            "[::1]:9464",
            "[::1]",
            "::1",
            "[fe80::1]",
            "localhost",
            "LocalHost:9464",
            "box.lan",
            "BOX.lan:9464",
        ] {
            assert!(policy().allows(ok), "{ok}");
        }
        for bad in [
            "evil.example",
            "evil.example:9464",
            "127.0.0.1.evil.example",
            "localhost.evil.example",
            "box.lan.evil.example",
            "[::1",
            "[::1]x",
            "[::1]:port",
            "box.lan:abc",
            "",
            "\u{e9}",
        ] {
            assert!(!policy().allows(bad), "{bad}");
        }
        // With no listen host named (a bare IP listen), only literals and localhost.
        assert!(!HostPolicy::default().allows("box.lan"));
        assert_eq!(host_part("[::]:9464"), Some("::"));
        assert_eq!(host_part("0.0.0.0:9464"), Some("0.0.0.0"));
        let mut q = r("GET", "/stats", None);
        q.host = Some("evil.example".into());
        assert_eq!(route(&q, None), Reply::Status(403));
        assert_eq!(
            route(&q, Some("t")),
            Reply::Status(401),
            "a token replaces the Host check"
        );
        let mut q = r("GET", "/health", None);
        q.host = None;
        assert_eq!(route(&q, None), Reply::Status(400), "HTTP/1.1 needs Host");
        q.http11 = false;
        assert_eq!(route(&q, None), Reply::Health, "HTTP/1.0 may omit it");
        q.http11 = true;
        assert_eq!(route(&q, Some("t")), Reply::Health);
    }

    #[test]
    fn non_ascii_headers_do_not_panic() {
        // The 7th byte falls inside a multibyte character.
        for v in [
            "b\u{e9}arer x",
            "Bear\u{e9}r",
            "Bearer\u{e9}x",
            "Beare\u{e9}",
            "\u{e9}",
            "bearer \u{e9}\u{e9}",
        ] {
            let text = format!("GET / HTTP/1.1\r\nHost: h\r\nAuthorization: {v}\r\n");
            assert!(parse_request(text.as_bytes()).is_ok(), "{v}");
        }
        let r = req("GET / HTTP/1.1\r\nAuthorization: Bearer t\u{f6}k\u{e9}n").unwrap();
        assert_eq!(r.bearer.as_deref(), Some("t\u{f6}k\u{e9}n"));
        let r = req("GET / HTTP/1.1\r\nAuthorization: b\u{e9}arer x").unwrap();
        assert!(r.bearer.is_none());
        // Raw bytes that are not UTF-8 are a bad request, not a panic.
        assert!(parse_request(b"GET / HTTP/1.1\r\nAuthorization: Bearer \xc3").is_err());
        assert!(parse_request(b"GET /\xe9 HTTP/1.1\r\n").is_err());
        assert!(parse_request(b"\xff").is_err());
        let r = req("GET /\u{e9}?q=\u{e9} HTTP/1.1\r\nHost: \u{e9}").unwrap();
        assert_eq!(r.path, "/\u{e9}");
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
        s.write_all(format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}\r\n").as_bytes())
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
        let live = crate::observe::LiveStats::default();
        let rows = std::iter::once((&model, None, &perf, &live, 0, 8192));
        ex.tick(rows, &[], Instant::now());
    }

    fn server() -> (Exporter, SocketAddr) {
        let ex = Exporter::open(None, None, Duration::ZERO, "live")
            .unwrap()
            .with_http("127.0.0.1:0", Some("tok".into()), &[])
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
        assert!(!head.to_ascii_lowercase().contains("access-control"));
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
    async fn no_cors_and_host_check_without_a_token() {
        let mut ex = Exporter::open(None, None, Duration::ZERO, "live")
            .unwrap()
            .with_http("127.0.0.1:0", None, &[])
            .unwrap();
        tick(&mut ex);
        let addr = ex.listen_info().unwrap().0;
        let (head, _) = get(addr, "/stats", None).await;
        assert!(head.starts_with("HTTP/1.1 200"));
        assert!(!head.to_ascii_lowercase().contains("access-control"));
        let raw = |host: &'static str| async move {
            let mut s = TcpStream::connect(addr).await.unwrap();
            s.write_all(format!("GET /stats HTTP/1.1\r\n{host}\r\n").as_bytes())
                .await
                .unwrap();
            let mut out = Vec::new();
            timeout(Duration::from_secs(5), s.read_to_end(&mut out))
                .await
                .unwrap()
                .unwrap();
            String::from_utf8(out).unwrap()
        };
        let evil = raw("Host: evil.example\r\n").await;
        assert!(evil.starts_with("HTTP/1.1 403"), "{evil}");
        assert!(!evil.to_ascii_lowercase().contains("access-control"));
        assert!(raw("Host: localhost:9464\r\n")
            .await
            .starts_with("HTTP/1.1 200"));
        assert!(raw("Host: [::1]:9464\r\n")
            .await
            .starts_with("HTTP/1.1 200"));
        assert!(raw("").await.starts_with("HTTP/1.1 400"));
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
    async fn http10_stream_is_bare_ndjson_until_close() {
        let (mut ex, addr) = server();
        tick(&mut ex);
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(b"GET /stream HTTP/1.0\r\nAuthorization: Bearer tok\r\n\r\n")
            .await
            .unwrap();
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
        assert!(!head.contains("Transfer-Encoding"), "{head}");
        tick(&mut ex);
        for seq in 0..2 {
            let mut l = String::new();
            timeout(Duration::from_secs(5), r.read_line(&mut l))
                .await
                .unwrap()
                .unwrap();
            let v: serde_json::Value = serde_json::from_str(&l).unwrap();
            assert_eq!(v["seq"], seq);
        }
        // The close ends the body; no closing chunk follows.
        ex.close().await;
        let mut rest = Vec::new();
        r.read_to_end(&mut rest).await.unwrap();
        assert!(rest.is_empty(), "{:?}", String::from_utf8_lossy(&rest));
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
            .unwrap();
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
        let first = HttpServer::bind("127.0.0.1:0", None, &[]).unwrap();
        let taken = first.addr().to_string();
        assert!(HttpServer::bind(&taken, None, &[]).is_err());
    }
}
