use std::path::Path;
use tracing_core::{Level, LevelFilter};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{registry, Layer};

/// This trait is used to install an application logger.
pub trait ApplicationLoggerInstaller {
    /// Install the application logger.
    fn install(&self) -> Result<(), String>;
}

/// This struct is used to install a local file application logger to output logs to a given file path.
pub struct FileApplicationLoggerInstaller {
    path: String,
}

impl FileApplicationLoggerInstaller {
    /// Create a new file application logger.
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl ApplicationLoggerInstaller for FileApplicationLoggerInstaller {
    fn install(&self) -> Result<(), String> {
        let path = Path::new(&self.path);
        let writer = tracing_appender::rolling::never(
            path.parent().unwrap_or_else(|| Path::new(".")),
            path.file_name()
                .unwrap_or_else(|| panic!("The path '{}' to point to a file.", self.path)),
        );
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(writer)
            .with_filter(LevelFilter::INFO)
            .with_filter(filter_fn(|m| {
                if let Some(path) = m.module_path() {
                    // The wgpu crate is logging too much, so we skip `info` level.
                    if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                        return false;
                    }
                }
                true
            }));

        if registry().with(layer).try_init().is_err() {
            return Err("Failed to install the file logger.".to_string());
        }

        let hook = std::panic::take_hook();
        let file_path: String = self.path.to_owned();

        std::panic::set_hook(Box::new(move |info| {
            log::error!("PANIC => {}", info.to_string());
            log::error!("{}", std::backtrace::Backtrace::force_capture());
            eprintln!(
                "=== PANIC ===\nA fatal error happened, you can check the experiment logs here => \
                    '{file_path}'\n============="
            );
            hook(info);
        }));

        Ok(())
    }
}
