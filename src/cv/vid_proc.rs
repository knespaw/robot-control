use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::{ClockTime, FlowError, FlowSuccess, MessageView, Pipeline, Sample, State};
use gstreamer_app::{AppSink, AppSinkCallbacks};
use tracing::{Level, debug, error, info, warn};

use super::utils::*;
use crate::utils::logging::log_message;



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
const PIPELINE_STR: &str =
	"avfvideosrc device-index=<D> ! \
	video/x-raw,width=<W1>,height=<H1>,framerate=<F>/1 ! \
	videoconvert ! \
	videoflip method=automatic ! \
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
		info!("initializing video stream");
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
		debug!(
			sink_name = name,
			device_idx,
			fps,
			capture_width = capture_res.0,
			capture_height = capture_res.1,
			target_width = target_res.0,
			target_height = target_res.1,
			"building video pipeline"
		);
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

		log_message(Level::DEBUG, || format!("launching GStreamer pipeline >> {}", pipeline_str));

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
				let sample = sink.pull_sample().map_err(|_| {
					error!("failed to pull sample from appsink");
					FlowError::Eos
				})?;

				match tx.try_send(sample)
				{
					Ok(true) => Ok(FlowSuccess::Ok),

					Ok(false) =>
					{
						warn!("dropping sample because processing channel is full");
						Ok(FlowSuccess::Ok)
					},

					Err(err) =>
					{
						error!(error = ?err, "failed to forward sample to processing channel");
						Err(FlowError::Eos)
					},
				}
			})
			.build();

		appsink.set_callbacks(func);
	}

	pub(crate) fn start(&self) -> CvResult<()>
	{
		info!("starting video pipeline");
		self.pipeline
			.set_state(State::Playing)?;

		let bus = self
			.pipeline
			.bus()
			.ok_or_else(|| CvError::PipelineBus)?;

		for msg in bus.iter_timed(ClockTime::NONE)
		{
			match msg.view()
			{
				MessageView::Error(e) =>
				{
					error!(
						error = %e.error(),
						debug = ?e.debug(),
						"video pipeline reported an error",
					);
					break;
				},

				MessageView::Eos(_) =>
				{
					info!("video pipeline reached EOS");
					break;
				},

				_ => (),
			}
		}

		info!("stopping video pipeline");
		self.pipeline.set_state(State::Null)?;

		Ok(())
	}
}



#[cfg(test)]
mod tests
{
	use std::path::{Path, PathBuf};
	use std::time::Duration;
	use std::{fs, thread};

	use image::RgbImage;
	use tracing_test::traced_test;

	use super::*;



	const TEST_IMAGES_DIR : &str = "assets/images";
	const TEST_CAMERA_DEVICE_IDX : u8 = 0;
	const TEST_CAPTURE_DURATION_SECS : u64 = 5;
	const TEST_CAPTURE_INTERVAL_MS : u64 = 500;



	fn output_image_path(
		images_dir : &Path,
		idx : usize,
	) -> PathBuf
	{
		images_dir.join(format!("camera_feed_{idx:02}.jpg"))
	}


	fn save_sample_as_image(
		sample : &Sample,
		path : &Path,
		width : u32,
		height : u32,
	)
	{
		let buffer = sample
			.buffer()
			.expect("sample should contain a buffer");
		let map = buffer
			.map_readable()
			.expect("sample buffer should be readable");
		let raw_rgb_data = map.as_slice().to_vec();

		let image = RgbImage::from_raw(width, height, raw_rgb_data)
			.expect("sample should contain a full RGB frame");

		image
			.save(path)
			.expect("captured frame should be written");
	}



	#[tokio::test]
	#[traced_test]
	#[cfg(target_os = "macos")]
	async fn test_basic_camera_pipeline()
	{
		gst::init().expect("GStreamer should initialize");

		let images_dir = Path::new(TEST_IMAGES_DIR);

		fs::create_dir_all(images_dir).expect("test images directory should exist");

		let (tx, rx) = kanal::bounded(32);
		let params = StreamParameters::default();
		let frame_width = params.target_width as u32;
		let frame_height = params.target_height as u32;

		let stream = Stream::new("test-camera-feed", TEST_CAMERA_DEVICE_IDX, params, tx)
			.expect("camera pipeline should initialize");

		let pipeline = stream.pipeline.clone();
		let handle = thread::spawn(move || stream.start());

		let n_images = (TEST_CAPTURE_DURATION_SECS * 1000 / TEST_CAPTURE_INTERVAL_MS) as usize;

		for idx in 0 .. n_images
		{
			tokio::time::sleep(Duration::from_millis(TEST_CAPTURE_INTERVAL_MS)).await;

			let mut sample = rx
				.recv_timeout(Duration::from_secs(2))
				.expect("camera stream should produce a frame");

			while let Ok(Some(newer_sample)) = rx.try_recv()
			{
				sample = newer_sample;
			}

			save_sample_as_image(
				&sample,
				&output_image_path(images_dir, idx),
				frame_width,
				frame_height,
			);
		}

		assert!(pipeline.send_event(gst::event::Eos::new()), "pipeline should accept EOS event");

		handle
			.join()
			.expect("camera pipeline thread should not panic")
			.expect("camera pipeline should stop cleanly");
	}
}
