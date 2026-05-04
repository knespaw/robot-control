use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer::{ClockTime, FlowError, FlowSuccess, MessageView, Pipeline, Sample, State};
use gstreamer_app::{AppSink, AppSinkCallbacks};

use super::utils::*;



#[rustfmt::skip]
const PIPELINE_STR : &str =
    "avfvideosrc ! \
    videorate ! \
    video/x-raw,framerate=<F>/1 ! \
    videoconvert ! \
    videoscale ! \
    video/x-raw,width=<W>,height=<H>,format=RGB ! \
    appsink name=<N>";



struct Stream
{
	pipeline : Pipeline,
	sink :     AppSink,
}

impl Stream
{
	fn build(
		name : &str,
		width : usize,
		height : usize,
		fps : usize,
		max_buffers : Option<usize>,
		tx : kanal::Sender<Sample>,
	) -> CvResult<Self>
	{
		let pipeline = Self::launch_pipeline(name, width, height, fps)?;

		let sink = Self::init_appsink(name, &pipeline, max_buffers)?;

		Self::set_callback(&sink, tx);

		Ok(Stream { pipeline, sink })
	}

	fn launch_pipeline(
		name : &str,
		width : usize,
		height : usize,
		fps : usize,
	) -> CvResult<Pipeline>
	{
		let pipeline_str = PIPELINE_STR
			.replace("<W>", &width.to_string())
			.replace("<H>", &height.to_string())
			.replace("<F>", &fps.to_string())
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
		max_buffers : Option<usize>,
	) -> CvResult<AppSink>
	{
		let appsink = pipeline
			.by_name(appsink_name)
			.ok_or_else(|| CvError::AppSink)?
			.downcast::<AppSink>()
			.map_err(|_| CvError::AppSink)?;

		if let Some(buff) = max_buffers
		{
			appsink.set_max_buffers(buff as u32);
			appsink.set_drop(true);
		}

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

					Err(err) =>
					{
						// TODO
						Err(FlowError::Eos)
					},
				}
			})
			.build();

		appsink.set_callbacks(func);
	}

	fn start(&self) -> CvResult<()>
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
				MessageView::Error(e) => break,

				MessageView::Eos(_) => break,

				_ => (),
			}
		}

		self.pipeline.set_state(State::Null)?;

		Ok(())
	}
}
