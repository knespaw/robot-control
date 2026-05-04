use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::{ClockTime, FlowError, FlowSuccess, MessageView, Pipeline, Sample, State};
use gstreamer_app::{AppSink, AppSinkCallbacks};

use super::utils::*;



#[rustfmt::skip]
/// GStreamer [`Pipeline`] used to capture the camera feed.
///
/// Components:
/// >
/// > * `avfvideosrc` - *AVFoundation* video source (0 - macOS built-in camera,
/// 1-2 - *Continuity Camera* feed)
/// >
/// > * `video/x-raw,width=<W1>,height=<H1>,framerate=<F>/1` - enforces the pipeline to capture
/// frames at a specified resolution and framerate
/// >
/// > * `add-borders=true` & `pixel-aspect-ratio=1/1` - the letterbox enablers; scale the capture
/// resolution down, but preserve the aspect ratio by adding extra black bars to the frame
/// >
/// > * `max-buffers=1` & `drop=true` - dropping the old frames (in case the [`AppSink`] callback
/// does not handle them fast enough) so that only the most recent capture is being processed
// TODO: check if `cvequalizehist !` will be enough
const PIPELINE_STR: &str =
	"avfvideosrc device-index=<D> ! \
	video/x-raw,width=<W1>,height=<H1>,framerate=<F>/1 ! \
	videoconvert ! \
	cvequalizehist ! \
	videoscale add-borders=true ! \
	video/x-raw,width=<W2>,height=<H2>,pixel-aspect-ratio=1/1,format=RGB ! \
	appsink name=<N> emit-signals=true max-buffers=1 drop=true";



pub(crate) struct StreamParameters
{
	fps :            u8,
	capture_width :  u16,
	capture_height : u16,
	target_width :   u16,
	target_height :  u16,
}

impl Default for StreamParameters
{
	fn default() -> Self
	{
		StreamParameters {
			fps :            30,
			capture_width :  1920,
			capture_height : 1080,
			target_width :   640,
			target_height :  640,
		}
	}
}



pub(crate) struct Stream
{
	pipeline : Pipeline,
	#[allow(dead_code)]
	sink :     AppSink,
}

impl Stream
{
	pub(crate) fn new(
		name : &str,
		device_idx : u8,
		params : StreamParameters,
		tx : kanal::Sender<Sample>,
	) -> CvResult<Self>
	{
		Self::build(
			name,
			device_idx,
			params.fps,
			(params.capture_width, params.capture_height),
			(params.target_width, params.target_height),
			tx,
		)
	}

	fn build(
		name : &str,
		device_idx : u8,
		fps : u8,
		capture_res : (u16, u16),
		target_res : (u16, u16),
		tx : kanal::Sender<Sample>,
	) -> CvResult<Self>
	{
		let pipeline = Self::launch_pipeline(name, device_idx, fps, capture_res, target_res)?;

		let sink = Self::init_appsink(name, &pipeline)?;

		Self::set_callback(&sink, tx);

		Ok(Stream { pipeline, sink })
	}

	fn launch_pipeline(
		name : &str,
		device_idx : u8,
		fps : u8,
		capture_res : (u16, u16),
		target_res : (u16, u16),
	) -> CvResult<Pipeline>
	{
		let pipeline_str = PIPELINE_STR
			.replace("<D>", &device_idx.to_string())
			.replace("<F>", &fps.to_string())
			.replace("<W1>", &capture_res.0.to_string())
			.replace("<H1>", &capture_res.1.to_string())
			.replace("<W2>", &target_res.0.to_string())
			.replace("<H2>", &target_res.1.to_string())
			.replace("<N>", name);

		let pipeline = gst::parse::launch(&pipeline_str)
			.map_err(|e| CvError::PipelineLaunch(Some(e)))?
			.downcast::<Pipeline>()
			.map_err(|_| CvError::PipelineLaunch(None))?;

		Ok(pipeline)
	}

	fn init_appsink(
		appsink_name : &str,
		pipeline : &Pipeline,
	) -> CvResult<AppSink>
	{
		let appsink = pipeline
			.by_name(appsink_name)
			.ok_or_else(|| CvError::AppSink)?
			.downcast::<AppSink>()
			.map_err(|_| CvError::AppSink)?;

		Ok(appsink)
	}

	fn set_callback(
		appsink : &AppSink,
		tx : kanal::Sender<Sample>,
	)
	{
		let func = AppSinkCallbacks::builder()
			.new_sample(move |sink| {
				// TODO: add logging here

				let sample = sink
					.pull_sample()
					.map_err(|_| FlowError::Eos)?;

				match tx.try_send(sample)
				{
					Ok(true) => Ok(FlowSuccess::Ok),

					Ok(false) =>
					{
						// TODO: failed to send the sample; add logging
						Ok(FlowSuccess::Ok)
					},

					Err(_err) =>
					{
						// TODO
						Err(FlowError::Eos)
					},
				}
			})
			.build();

		appsink.set_callbacks(func);
	}

	pub(crate) fn start(&self) -> CvResult<()>
	{
		self.pipeline
			.set_state(State::Playing)?;

		let bus = self
			.pipeline
			.bus()
			.ok_or_else(|| CvError::PipelineBus)?;

		// TODO: add logging

		for msg in bus.iter_timed(ClockTime::NONE)
		{
			match msg.view()
			{
				MessageView::Error(_e) => break,

				MessageView::Eos(_) => break,

				_ => (),
			}
		}

		self.pipeline.set_state(State::Null)?;

		Ok(())
	}
}
