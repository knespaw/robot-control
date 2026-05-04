#[derive(Debug, thiserror::Error)]
pub(crate) enum ComError
{
	#[error("Bluetooth communication initialization failed >> {0}")]
	BLEStart(String),

	#[error("Bluetooth data write failed >> {0}")]
	BLEWrite(btleplug::Error),
}


pub(crate) type ComResult<T> = Result<T, ComError>;
