use gstreamer::{StateChangeError, glib};



#[derive(Debug, thiserror::Error)]
pub(crate) enum CvError
{
	#[error(
        "failed to start GStreamer pipeline >> {}",
        .0
        .as_ref()
        .map(|e| format!(" >> {}", e))
        .unwrap_or("failed to downcast the element".into()),
    )]
	PipelineLaunch(Option<glib::Error>),

	#[error("failed to change GStreamer pipeline state >> {0}")]
	PipelineStart(#[from] StateChangeError),

	#[error("failed to get GStreamer pipeline bus")]
	PipelineBus,

	#[error("failed to get GStreamer pipeline AppSink")]
	AppSink,

	#[error("failed to initialize ArUco marker detector >> {0}")]
	MarkerInit(opencv::Error),

	#[error("failed to update ArUco marker detector image data >> {0}")]
	MarkerUpdate(opencv::Error),

	#[error("ArUco marker detection failed >> {0}")]
	MarkerDetection(opencv::Error),
}


pub(crate) type CvResult<T> = Result<T, CvError>;
