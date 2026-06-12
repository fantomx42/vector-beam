//! Command-line parsing for vector-beam.
//!
//! Flags are plain `--name value` pairs parsed by hand — the set is small
//! enough that a dependency isn't warranted, but the `Option` semantics
//! (e.g. "was `--persistence` given at all?") and the growing flag count
//! justify keeping the logic out of `main` and under test.

use crate::geometry::Scene;

/// User-requested present mode (`--present-mode`). `None` in [`Cli`] means
/// auto-select: the host tries Immediate, then Mailbox, then Fifo, taking the
/// first the surface supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentModeArg {
    Immediate,
    Mailbox,
    Fifo,
}

impl PresentModeArg {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "immediate" => Some(Self::Immediate),
            "mailbox" => Some(Self::Mailbox),
            "fifo" => Some(Self::Fifo),
            _ => None,
        }
    }

    pub fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::Immediate => wgpu::PresentMode::Immediate,
            Self::Mailbox => wgpu::PresentMode::Mailbox,
            Self::Fifo => wgpu::PresentMode::Fifo,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Screenshot {
    pub path: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct Cli {
    /// `None` = flag absent, so the mode-dependent default applies.
    pub persistence: Option<f32>,
    pub scene: Scene,
    pub screenshot: Option<Screenshot>,
    pub present_mode: Option<PresentModeArg>,
    pub fullscreen: bool,
}

/// Parse `args` (including `argv[0]`). Errors are user-facing messages for
/// stderr.
pub fn parse(args: &[String]) -> Result<Cli, String> {
    let persistence = match flag_value(args, "--persistence") {
        None => None,
        Some(v) => Some(
            v.parse::<f32>()
                .map_err(|_| format!("--persistence expects seconds (got {v:?})"))?
                .max(0.0),
        ),
    };

    let scene = match flag_value(args, "--scene") {
        None => Scene::default(),
        Some(name) => Scene::parse(name)
            .ok_or_else(|| format!("--scene expects one of: cube, lissajous (got {name:?})"))?,
    };

    let present_mode = match flag_value(args, "--present-mode") {
        None => None,
        Some(name) => Some(PresentModeArg::parse(name).ok_or_else(|| {
            format!("--present-mode expects one of: immediate, mailbox, fifo (got {name:?})")
        })?),
    };

    // `--screenshot [path] [WxH]`: both positionals optional, in either order
    // (a WxH-shaped arg is a size, anything else non-flag is the path).
    let screenshot = args.iter().position(|a| a == "--screenshot").map(|pos| {
        let mut path = None;
        let mut size = None;
        for a in args[pos + 1..].iter().take(2) {
            if a.starts_with("--") {
                break;
            }
            if let Some(wh) = parse_size(a) {
                size = Some(wh);
            } else if path.is_none() {
                path = Some(a.clone());
            }
        }
        let (width, height) = size.unwrap_or((1280, 960));
        Screenshot {
            path: path.unwrap_or_else(|| "docs/screenshot.png".to_string()),
            width,
            height,
        }
    });

    Ok(Cli {
        persistence,
        scene,
        screenshot,
        present_mode,
        fullscreen: flag_present(args, "--fullscreen"),
    })
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|p| args.get(p + 1))
        .map(String::as_str)
}

fn flag_present(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        std::iter::once("vector-beam")
            .chain(s.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn persistence_absent_is_none() {
        assert_eq!(parse(&args(&[])).unwrap().persistence, None);
    }

    #[test]
    fn persistence_given_is_some_and_clamped() {
        assert_eq!(
            parse(&args(&["--persistence", "0.25"])).unwrap().persistence,
            Some(0.25)
        );
        assert_eq!(
            parse(&args(&["--persistence", "-1"])).unwrap().persistence,
            Some(0.0)
        );
    }

    #[test]
    fn persistence_bad_value_errors() {
        assert!(parse(&args(&["--persistence", "fast"])).is_err());
        assert!(parse(&args(&["--persistence"])).unwrap().persistence.is_none());
    }

    #[test]
    fn scene_parses_and_rejects_unknown() {
        assert_eq!(
            parse(&args(&["--scene", "lissajous"])).unwrap().scene,
            Scene::Lissajous
        );
        assert!(parse(&args(&["--scene", "teapot"])).is_err());
    }

    #[test]
    fn present_mode_parses_and_rejects_unknown() {
        assert_eq!(
            parse(&args(&["--present-mode", "mailbox"])).unwrap().present_mode,
            Some(PresentModeArg::Mailbox)
        );
        assert!(parse(&args(&["--present-mode", "vsync"])).is_err());
    }

    #[test]
    fn screenshot_positional_defaults() {
        let shot = |s: &[&str]| parse(&args(s)).unwrap().screenshot;
        assert_eq!(shot(&[]), None);
        assert_eq!(
            shot(&["--screenshot"]),
            Some(Screenshot { path: "docs/screenshot.png".into(), width: 1280, height: 960 })
        );
        assert_eq!(
            shot(&["--screenshot", "out.png", "640x480"]),
            Some(Screenshot { path: "out.png".into(), width: 640, height: 480 })
        );
        // Size without a path: WxH-shaped args are sizes, not paths.
        assert_eq!(
            shot(&["--screenshot", "640x480"]),
            Some(Screenshot { path: "docs/screenshot.png".into(), width: 640, height: 480 })
        );
        // A following flag is not a positional.
        assert_eq!(
            shot(&["--screenshot", "--fullscreen"]),
            Some(Screenshot { path: "docs/screenshot.png".into(), width: 1280, height: 960 })
        );
    }

    #[test]
    fn fullscreen_flag() {
        assert!(!parse(&args(&[])).unwrap().fullscreen);
        assert!(parse(&args(&["--fullscreen"])).unwrap().fullscreen);
    }
}
