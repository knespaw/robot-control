use gstreamer::Sample;
use kanal::{AsyncSender, Receiver};
use rayon::prelude::*;

use super::engine::*;
use super::utils::*;
use crate::com::Pool;
use crate::cv::{Detection, convert_image};



const MODEL_PATH : &str = "yolo/yolov8m-worldv2.onnx";


const INP_KEY : &str = "images";
const INP_CHANNELS : usize = 3;
pub(crate) const INP_HEIGHT : usize = 640;
pub(crate) const INP_WIDTH : usize = 640;
const INP_SHAPE : [usize; 4] = [1, INP_CHANNELS, INP_HEIGHT, INP_WIDTH];


const OUT_KEY : &str = "output0";
const OUT_SHAPE : [usize; 3] = [1, 7, 8400];


const LABELS : (&str, &str, &str) = ("small tracked robot", "battery pack", "circuit board");



struct Predictor
{
	engine :           Engine,
	img_rx :           Receiver<Sample>,
	predictions_pool : Pool<Vec<f32>>,
	_inp :             TensorSymbol<f32>,
}

impl Predictor
{
	fn init(
		img_rx : Receiver<Sample>,
		predictions_pool : Pool<Vec<f32>>,
	) -> MlResult<Self>
	{
		let mut _inp = TensorSymbol::<f32>::new(INP_KEY, INP_SHAPE)?;

		let engine = Engine::start(
			MODEL_PATH,
			true,
			Default::default(),
			Default::default(),
			vec![TensorSymbol::<f32>::new(OUT_KEY, OUT_SHAPE)?],
			vec![&mut _inp],
		)?;

		Ok(Predictor { engine, img_rx, _inp, predictions_pool })
	}

	fn run_stream(&mut self)
	{
		while let Ok(sample) = self.img_rx.recv()
		{
			self.process_stream_sample(sample);

			let mut outputs = self.engine.infer().unwrap(); // TODO

			// must be present
			let boxes_value = unsafe {
				outputs
					.remove(OUT_KEY)
					.unwrap_unchecked()
			};

			// must be present
			let boxes_data : &[f32] = unsafe {
				boxes_value
					.try_extract_tensor::<f32>()
					.unwrap_unchecked()
					.1
			};

			let buffer = match self.predictions_pool.rx().try_recv()
			{
				Ok(Some(mut buf)) =>
				{
					buf.clear();
					buf.extend_from_slice(boxes_data);
					buf
				},

				Ok(None) => boxes_data.to_vec(),

				Err(_) => panic!("should not happen"), // TODO
			};

			match self
				.predictions_pool
				.tx()
				.try_send(buffer)
			{
				Ok(true) => (),

				Ok(false) => (),

				Err(_) => panic!("should not happen"), // TODO
			}
		}
	}

	fn process_stream_sample(
		&mut self,
		sample : Sample,
	)
	{
		if let Some(buffer) = sample.buffer()
			&& let Ok(map) = buffer.map_readable()
		{
			let raw_rgb_data = map.as_slice();

			let tensor_data = unsafe { self._inp.held_data() };

			convert_image::<INP_HEIGHT, INP_WIDTH, INP_CHANNELS>(raw_rgb_data, tensor_data);
		}
	}
}



struct Postprocessor
{
	confidence_threshold : f32,
	iou_threshold :        f32,
	predictions_pool :     Pool<Vec<f32>>,
	detection_tx :         AsyncSender<InferenceResult>,
}

impl Postprocessor
{
	fn process_raw_detection(
		threshold : f32,
		data : &[f32; OUT_SHAPE[1]],
	) -> Option<Detection<INP_HEIGHT, INP_WIDTH>>
	{
		let (mut max_score, mut label) = (data[4], LABELS.0);
		if data[5] > max_score
		{
			max_score = data[5];
			label = LABELS.1;
		}
		if data[6] > max_score
		{
			max_score = data[6];
			label = LABELS.2;
		}

		if max_score >= threshold
		{
			let (xc, yc, width, height) = (data[0], data[1], data[2], data[3]);

			Some(Detection::from_coordinates(label, max_score, xc, yc, width, height))
		}
		else
		{
			None
		}
	}

	fn filter_detections(
		&mut self,
		raw_detections : &[f32],
	) -> InferenceResult
	{
		let mut filtered_detections = [None, None, None];

		raw_detections
			.par_chunks_exact(OUT_SHAPE[1])
			.filter_map(|raw_detection| {
				Self::process_raw_detection(
					self.confidence_threshold,
					raw_detection
						.try_into()
						.expect("should have exactly 7 elements"),
				)
			})
			.fold(
				|| [Detection::default(), Detection::default(), Detection::default()],
				|mut acc, det| {
					let idx = match det.label
					{
						l if l == LABELS.0 => 0,
						l if l == LABELS.1 => 1,
						_ => 2,
					};

					if det.confidence > acc[idx].confidence
					{
						acc[idx] = det;
					}

					acc
				},
			)
			.reduce(
				|| [Detection::default(), Detection::default(), Detection::default()],
				|mut acc_a, acc_b| {
					for idx in 0 .. 3
					{
						if acc_b[idx].confidence > acc_a[idx].confidence
						{
							acc_a[idx] = acc_b[idx];
						}
					}

					acc_a
				},
			)
			.into_iter()
			.enumerate()
			.for_each(|(idx, new_detection)| {
				// empty string is a default value, thus no detection was made
				if !new_detection.label.is_empty()
				{
					filtered_detections[idx] = Some(new_detection);
				}
			});

		filtered_detections
	}

	async fn run(&mut self)
	{
		while let Ok(buffer) = self
			.predictions_pool
			.rx()
			.as_async()
			.recv()
			.await
		{
			let final_detections = self.filter_detections(&buffer);

			// TODO
			let _ = self
				.predictions_pool
				.tx()
				.try_send(buffer);

			// TODO
			let _ = self
				.detection_tx
				.send(final_detections)
				.await;
		}
	}
}



pub(crate) type InferenceResult = [Option<Detection<INP_HEIGHT, INP_WIDTH>>; 3];
