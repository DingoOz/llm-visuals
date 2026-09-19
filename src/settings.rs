//! Saved launch settings and the settings screen (`s`).
//!
//! Saved settings are a small JSON map of flag name → value. At launch they
//! are turned back into flags and placed ahead of the real command line, so
//! clap validates them exactly like typed flags and anything given on the
//! command line still wins.

use crate::colors::THEME_NAMES;
use crate::config::Args;
use clap::Parser;
use crossterm::event::{KeyCode, KeyEvent};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;

const APP: &str = "llm-visuals";

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn home() -> Option<PathBuf> {
    env_dir("HOME").or_else(|| env_dir("USERPROFILE"))
}

/// Per-user directory for the saved settings.
pub fn config_dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        env_dir("APPDATA")
    } else if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library/Application Support"))
    } else {
        env_dir("XDG_CONFIG_HOME").or_else(|| home().map(|h| h.join(".config")))
    };
    base.map(|b| b.join(APP))
}

/// Per-user directory for data the dashboard writes, such as the log database.
pub fn data_dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        env_dir("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library/Application Support"))
    } else {
        env_dir("XDG_DATA_HOME").or_else(|| home().map(|h| h.join(".local/share")))
    };
    base.map(|b| b.join(APP))
}

pub fn config_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("settings.json"))
}

/// Where `--log-db auto` writes.
pub fn default_db_path() -> Option<PathBuf> {
    data_dir().map(|d| d.join("llm.db"))
}

/// Flag name (without dashes) → value.
type Saved = BTreeMap<String, String>;

fn load_saved() -> Result<Saved, String> {
    let Some(path) = config_path() else {
        return Ok(Saved::new());
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Saved::new()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// `--flag=value` keeps a value that starts with a dash from reading as a flag.
fn to_flags<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Vec<OsString> {
    pairs
        .into_iter()
        .map(|(k, v)| format!("--{k}={v}").into())
        .collect()
}

/// clap's first line, without its `error: ` prefix.
fn clap_message(e: clap::Error) -> String {
    let text = e.to_string();
    let line = text.lines().next().unwrap_or_default();
    line.strip_prefix("error: ").unwrap_or(line).to_string()
}

/// The parsed arguments plus the argv they came from, which the settings
/// screen re-parses with its own flags appended.
pub struct Launch {
    pub args: Args,
    pub argv: Vec<OsString>,
    /// Why the saved settings were not used, if they were not.
    pub warning: Option<String>,
}

pub fn launch() -> Launch {
    let cli: Vec<OsString> = std::env::args_os().collect();
    let saved = match load_saved() {
        Ok(saved) => saved,
        Err(e) => {
            return Launch {
                args: Args::parse_from(&cli),
                argv: cli,
                warning: Some(format!("saved settings not loaded: {e}")),
            }
        }
    };
    let mut warning = None;
    if !saved.is_empty() {
        let mut argv = cli[..1].to_vec();
        argv.extend(to_flags(
            saved.iter().map(|(k, v)| (k.as_str(), v.as_str())),
        ));
        argv.extend_from_slice(&cli[1..]);
        match Args::try_parse_from(&argv) {
            Ok(args) => {
                return Launch {
                    args,
                    argv,
                    warning,
                }
            }
            // A bad saved value must not stop the dashboard starting; when
            // the command line alone also fails, clap reports that below.
            Err(e) if Args::try_parse_from(&cli).is_ok() => {
                warning = Some(format!("saved settings ignored: {}", clap_message(e)));
            }
            Err(_) => {}
        }
    }
    Launch {
        args: Args::parse_from(&cli),
        argv: cli,
        warning,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Choice(&'static [&'static str]),
    Number,
    Text,
}

/// The logging switch is not a flag of its own: off writes `--log-db=off`.
const LOG_SWITCH: &str = "log";
const LOG_FILE: &str = "log-db";

pub struct Field {
    pub label: &'static str,
    /// Command-line flag without the dashes.
    flag: &'static str,
    pub kind: Kind,
    pub value: String,
    /// Value when the screen opened, to mark what was edited.
    initial: String,
    pub help: String,
    /// Takes effect only at the next launch.
    pub next_launch: bool,
}

impl Field {
    pub fn edited(&self) -> bool {
        self.value != self.initial
    }
}

pub enum Action {
    None,
    Close,
    /// Use the values for this session.
    Apply,
    /// Use them for this session and write them as the launch defaults.
    Save,
}

pub struct SettingsForm {
    pub fields: Vec<Field>,
    pub selected: usize,
    /// Text being typed into the selected field.
    pub editing: Option<String>,
    /// Why the last apply or save was refused.
    pub error: Option<String>,
}

impl SettingsForm {
    pub fn new(args: &Args) -> Self {
        let logging = !matches!(args.log_db.as_str(), "off" | "none" | "");
        let file = if logging {
            args.log_db.clone()
        } else {
            "auto".into()
        };
        let auto_path = default_db_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "no home directory".into());
        let field = |label, flag, kind, value: String, help: String| Field {
            label,
            flag,
            kind,
            initial: value.clone(),
            value,
            help,
            next_launch: false,
        };
        let mut fields = vec![
            field(
                "Theme",
                "theme",
                Kind::Choice(THEME_NAMES),
                args.theme.clone(),
                "Colour theme for the heat maps and tiles (also cycled by t)".into(),
            ),
            field(
                "Colour depth",
                "color",
                Kind::Choice(&["auto", "truecolor", "256"]),
                args.color.clone(),
                "auto detects truecolor from the terminal; force it if colours look flat".into(),
            ),
            field(
                "Poll interval (ms)",
                "poll-ms",
                Kind::Number,
                args.poll_ms.to_string(),
                "How often each server is sampled; 50 ms minimum".into(),
            ),
            field(
                "Max models",
                "max-models",
                Kind::Number,
                args.max_models.to_string(),
                "Most detected models to watch at once".into(),
            ),
            field(
                "Watch PIDs",
                "pid",
                Kind::Text,
                args.pid.clone(),
                "all, or a comma-separated list of process IDs".into(),
            ),
            field(
                "GPUs",
                "gpu",
                Kind::Text,
                args.gpu.clone(),
                "all, or comma-separated GPU indices, e.g. 0,1".into(),
            ),
            field(
                "SQLite logging",
                LOG_SWITCH,
                Kind::Choice(&["on", "off"]),
                if logging { "on" } else { "off" }.into(),
                "Record model/GPU samples and finished requests to a SQLite file \
                 (auto stays off in --demo)"
                    .into(),
            ),
            field(
                "Database file",
                LOG_FILE,
                Kind::Text,
                file,
                format!("auto = {auto_path}, or a path of your own"),
            ),
            field(
                "Log interval (s)",
                "log-every",
                Kind::Number,
                args.log_every.to_string(),
                "Seconds between sample rows".into(),
            ),
            field(
                "Database cap (MB)",
                "log-db-max-mb",
                Kind::Number,
                args.log_db_max_mb.to_string(),
                "Oldest rows are dropped past this size; 0 = no cap".into(),
            ),
            field(
                "Max layers",
                "max-layers",
                Kind::Number,
                args.max_layers.to_string(),
                "Layers shown in the attention view; 0 = all".into(),
            ),
            field(
                "Max heads",
                "max-heads",
                Kind::Number,
                args.max_heads.to_string(),
                "Heads per layer shown in the attention view; 0 = all".into(),
            ),
        ];
        // The GPU collector is spawned once with its filter.
        if let Some(f) = fields.iter_mut().find(|f| f.flag == "gpu") {
            f.next_launch = true;
        }
        Self {
            fields,
            selected: 0,
            editing: None,
            error: None,
        }
    }

    fn value(&self, flag: &str) -> &str {
        self.fields
            .iter()
            .find(|f| f.flag == flag)
            .map(|f| f.value.as_str())
            .unwrap_or_default()
    }

    /// The flags that reproduce what the screen shows.
    fn flags(&self) -> Vec<(&'static str, String)> {
        let logging = self.value(LOG_SWITCH) != "off";
        self.fields
            .iter()
            .filter(|f| f.flag != LOG_SWITCH)
            .map(|f| match f.flag {
                LOG_FILE if !logging => (f.flag, "off".to_string()),
                LOG_FILE if f.value.is_empty() => (f.flag, "auto".to_string()),
                _ => (f.flag, f.value.clone()),
            })
            .collect()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        self.error = None;
        let field = &mut self.fields[self.selected];
        let kind = field.kind;
        if let Some(buf) = self.editing.as_mut() {
            match key.code {
                KeyCode::Char(c) => buf.push(c),
                KeyCode::Backspace => {
                    buf.pop();
                }
                KeyCode::Enter => {
                    field.value = buf.trim().to_string();
                    self.editing = None;
                }
                KeyCode::Esc => self.editing = None,
                _ => {}
            }
            return Action::None;
        }
        let n = self.fields.len();
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('s') => return Action::Close,
            KeyCode::Up | KeyCode::Char('k') => self.selected = (self.selected + n - 1) % n,
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.selected = (self.selected + 1) % n
            }
            KeyCode::Left => self.cycle(-1),
            KeyCode::Right => self.cycle(1),
            KeyCode::Enter | KeyCode::Char(' ') => match kind {
                Kind::Choice(_) => self.cycle(1),
                _ => self.editing = Some(self.fields[self.selected].value.clone()),
            },
            KeyCode::Char('d') => {
                let defaults = SettingsForm::new(&Args::parse_from([APP]));
                self.fields[self.selected].value = defaults.fields[self.selected].value.clone();
            }
            KeyCode::Char('a') => return Action::Apply,
            KeyCode::Char('w') => return Action::Save,
            _ => {}
        }
        Action::None
    }

    fn cycle(&mut self, step: isize) {
        let field = &mut self.fields[self.selected];
        if let Kind::Choice(options) = field.kind {
            let n = options.len() as isize;
            let i = options.iter().position(|o| *o == field.value).unwrap_or(0) as isize;
            field.value = options[(i + step).rem_euclid(n) as usize].to_string();
        }
    }

    /// The arguments this session would have with the screen's values, as
    /// clap parses them on top of the launch argv (so errors read the same
    /// as on the command line).
    pub fn resolve(&self, launch_argv: &[OsString]) -> Result<Args, String> {
        let flags = self.flags();
        let mut argv = launch_argv.to_vec();
        argv.extend(to_flags(flags.iter().map(|(k, v)| (*k, v.as_str()))));
        Args::try_parse_from(argv).map_err(clap_message)
    }

    /// Write the screen's values as the launch defaults, keeping any other
    /// saved flags.
    pub fn save(&self) -> Result<PathBuf, String> {
        let path = config_path().ok_or("no home directory to save settings in")?;
        let defaults = SettingsForm::new(&Args::parse_from([APP])).flags();
        let mut saved = load_saved().unwrap_or_default();
        merge(&mut saved, self.flags(), defaults);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(&saved).map_err(|e| e.to_string())?;
        std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }
}

/// Values equal to the built-in default are dropped rather than pinned, so a
/// later change of default still reaches this user.
fn merge(
    saved: &mut Saved,
    flags: Vec<(&'static str, String)>,
    defaults: Vec<(&'static str, String)>,
) {
    for ((flag, value), (_, default)) in flags.into_iter().zip(defaults) {
        if value == default {
            saved.remove(flag);
        } else {
            saved.insert(flag.to_string(), value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(a: &[&str]) -> Vec<OsString> {
        a.iter().map(OsString::from).collect()
    }

    #[test]
    fn command_line_overrides_saved_flags() {
        let a = Args::try_parse_from(argv(&[
            APP,
            "--theme=neon",
            "--poll-ms=500",
            "--theme",
            "fire",
        ]))
        .unwrap();
        assert_eq!(a.theme, "fire");
        assert_eq!(a.poll_ms, 500);
    }

    #[test]
    fn logging_is_on_by_default_except_in_demo() {
        let a = Args::try_parse_from([APP]).unwrap();
        assert_eq!(a.log_db, "auto");
        assert_eq!(a.log_db_path(), default_db_path());
        assert!(Args::try_parse_from([APP, "--demo"])
            .unwrap()
            .log_db_path()
            .is_none());
        assert!(Args::try_parse_from([APP, "--log-db", "off"])
            .unwrap()
            .log_db_path()
            .is_none());
        let explicit = Args::try_parse_from([APP, "--demo", "--log-db", "x.db"]).unwrap();
        assert_eq!(explicit.log_db_path(), Some(PathBuf::from("x.db")));
    }

    #[test]
    fn form_round_trips_and_switches_logging_off() {
        let launch = argv(&[APP, "--theme", "ocean", "--log-db", "my.db"]);
        let args = Args::try_parse_from(&launch).unwrap();
        let mut form = SettingsForm::new(&args);
        let same = form.resolve(&launch).unwrap();
        assert_eq!(same.theme, "ocean");
        assert_eq!(same.log_db, "my.db");

        form.selected = form
            .fields
            .iter()
            .position(|f| f.flag == LOG_SWITCH)
            .unwrap();
        form.cycle(1);
        let off = form.resolve(&launch).unwrap();
        assert!(off.log_db_path().is_none());
    }

    #[test]
    fn bad_value_is_reported_not_applied() {
        let launch = argv(&[APP]);
        let mut form = SettingsForm::new(&Args::try_parse_from(&launch).unwrap());
        let i = form
            .fields
            .iter()
            .position(|f| f.flag == "poll-ms")
            .unwrap();
        form.fields[i].value = "fast".into();
        let err = form.resolve(&launch).unwrap_err();
        assert!(err.contains("poll-ms"), "{err}");
    }

    #[test]
    fn saving_keeps_other_flags_and_drops_defaults() {
        let defaults = SettingsForm::new(&Args::parse_from([APP]));
        let mut form = SettingsForm::new(&Args::parse_from([APP, "--theme", "fire"]));
        let mut saved = Saved::from([
            ("poll-ms".to_string(), "900".to_string()),
            ("model".to_string(), "auto".to_string()),
        ]);
        // The screen shows the default poll interval, so the saved 900 goes.
        form.fields[0].value = "neon".into();
        merge(&mut saved, form.flags(), defaults.flags());
        assert_eq!(saved.get("theme").map(String::as_str), Some("neon"));
        assert!(!saved.contains_key("poll-ms"));
        assert_eq!(saved.get("model").map(String::as_str), Some("auto"));
    }
}
