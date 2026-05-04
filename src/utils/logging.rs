use tracing::{Level, debug, enabled, error, info, trace, warn};



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
