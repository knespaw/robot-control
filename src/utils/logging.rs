use std::sync::Once;

use tracing::{Level, debug, enabled, error, info, trace, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;



static LOGGING_INIT : Once = Once::new();



pub(crate) fn init_logging()
{
	LOGGING_INIT.call_once(|| {
		let env_filter =
			EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

		tracing_subscriber::registry()
			.with(env_filter)
			.with(tracing_subscriber::fmt::layer().compact())
			.init();

		info!("tracing initialized");
	});
}



fn check_log_level(level : Level) -> bool
{
	match level
	{
		Level::TRACE => enabled!(Level::TRACE),
		Level::DEBUG => enabled!(Level::DEBUG),
		Level::INFO => enabled!(Level::INFO),
		Level::WARN => enabled!(Level::WARN),
		Level::ERROR => enabled!(Level::ERROR),
	}
}



pub(crate) fn log_message<F>(
	level : Level,
	msg_constructor : F,
) where
	F : FnOnce() -> String,
{
	if !check_log_level(level)
	{
		return;
	}

	let msg = msg_constructor();

	match level
	{
		Level::TRACE => trace!("{}", msg),
		Level::DEBUG => debug!("{}", msg),
		Level::INFO => info!("{}", msg),
		Level::WARN => warn!("{}", msg),
		Level::ERROR => error!("{}", msg),
	}
}
