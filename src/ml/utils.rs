#[derive(Debug, thiserror::Error)]
pub(crate) enum MlError
{
	#[error("inference engine initialization failed >> {0}")]
	Init(String),

	#[error("session build failed >> {0}")]
	SessionBuild(String),

	#[error("model ONNX file either is corrupted, or does not exist >> {0}")]
	ModelFile(String),

	#[error("model compilation failed >> {0}")]
	Compilation(ort::Error),

	#[error("IO binding failed >> {0}")]
	Binding(ort::Error),

	#[error("failed to synchronize bound inputs >> {0}")]
	Sync(ort::Error),

	#[error("inference failed >> {0}")]
	Inference(ort::Error),

	#[error("tensor error >> {0}")]
	Tensor(String),
}


pub(crate) type MlResult<T> = Result<T, MlError>;
