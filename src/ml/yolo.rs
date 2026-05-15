use std::sync::atomic::{AtomicUsize, Ordering};

use gstreamer::Sample;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use kanal::{AsyncSender, Receiver};
use rayon::prelude::*;
use tracing::{debug, error, info, warn};

use super::engine::*;
use super::utils::*;
use crate::com::Pool;
use crate::cv::convert_image;
use crate::cv::obj_detect::{Corners, Detection, MarkerDetector, Object};
use crate::utils::comp::sigmoid;



const MODEL_PATH : &str = "yolo/yolov8s-worldv2.onnx";


const INP_KEY : &str = "images";
const INP_CHANNELS : u16 = 3;
pub(crate) const INP_HEIGHT : u16 = 640;
pub(crate) const INP_WIDTH : u16 = 640;
const INP_SHAPE : (u16, u16, u16, u16) = (1, INP_CHANNELS, INP_HEIGHT, INP_WIDTH);


const OUT_KEY : &str = "output0";
const OUT_SHAPE : (u16, u16, u16) = (1, 7, 8400);

// TEMPORARY DEBUG START: save annotated postprocessed frames.
const DEBUG_SAVE_DETECTIONS : bool = true;
const DEBUG_SAVE_EVERY_N_FRAMES : usize = 15;
const DEBUG_SAVE_DIR : &str = "assets/images/yolo-debug";
static DEBUG_SAVE_COUNTER : AtomicUsize = AtomicUsize::new(0);
// TEMPORARY DEBUG END: save annotated postprocessed frames.


pub(crate) const TRACKED : Object = Object {
	min_size :             10.0,
	max_size :             300.0,
	confidence_threshold : 0.5,
	iou_threshold :        0.5,
	label :                "robot with rubber tracks",
};
pub(crate) const TARGET : Object = Object {
	min_size :             10.0,
	max_size :             100.0,
	confidence_threshold : 0.5,
	iou_threshold :        0.5,
	label :                "yellow small circular block",
};
pub(crate) const OBSTACLE : Object = Object {
	min_size :             10.0,
	max_size :             100.0,
	confidence_threshold : 0.5,
	iou_threshold :        0.5,
	label :                "green rectangular block",
};



#[derive(Default)]
pub(crate) struct Prediction
{
	model_detections :    Vec<f32>,
	reference_detection : Option<Corners>,
	debug_frame :         Option<Vec<u8>>,
}



pub(crate) struct Predictor
{
	engine :           Engine,
	img_rx :           Receiver<Sample>,
	ref_detector :     MarkerDetector<INP_HEIGHT, INP_WIDTH>,
	predictions_pool : Pool<Prediction>,
	_inp :             TensorSymbol<f32>,
}

impl Predictor
{
	pub(crate) fn init(
		img_rx : Receiver<Sample>,
		ref_detector : MarkerDetector<INP_HEIGHT, INP_WIDTH>,
		predictions_pool : Pool<Prediction>,
	) -> MlResult<Self>
	{
		info!("initializing YOLO-World predictor");

		let mut _inp = TensorSymbol::<f32>::new(
			INP_KEY,
			[
				INP_SHAPE.0 as usize,
				INP_SHAPE.1 as usize,
				INP_SHAPE.2 as usize,
				INP_SHAPE.3 as usize,
			],
		)?;

		let engine = Engine::start(
			MODEL_PATH,
			Default::default(),
			Default::default(),
			vec![TensorSymbol::<f32>::new(
				OUT_KEY,
				[OUT_SHAPE.0 as usize, OUT_SHAPE.1 as usize, OUT_SHAPE.2 as usize],
			)?],
			vec![&mut _inp],
		)?;

		Ok(Predictor { engine, img_rx, _inp, predictions_pool, ref_detector })
	}

	pub(crate) fn run_stream(&mut self)
	{
		info!("predictor stream loop started");
		while let Ok(sample) = self.img_rx.recv()
		{
			debug!("received sample for inference");

			let debug_frame = match self.process_stream_sample(sample)
			{
				Ok(debug_frame) => debug_frame,

				Err(e) =>
				{
					error!(error = %e, "failed to pre-process stream frame");
					continue;
				},
			};

			let mut outputs = match self.engine.infer()
			{
				Ok(outputs) => outputs,

				Err(e) =>
				{
					error!(error = %e, "inference failed for current sample");
					continue;
				},
			};

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

			let corners = self
				.ref_detector
				.detect()
				.unwrap_or_else(|_| {
					warn!("could not find reference object");
					None
				});

			let buffer = match self.predictions_pool.rx().try_recv()
			{
				Ok(Some(mut buf)) =>
				{
					buf.model_detections.clear();
					buf.model_detections
						.extend_from_slice(boxes_data);

					buf.reference_detection = corners;
					buf.debug_frame = debug_frame;

					buf
				},

				Ok(None) =>
				{
					debug!("prediction pool receiver returned no reusable buffer");

					Prediction {
						model_detections : boxes_data.to_vec(),
						reference_detection : corners,
						debug_frame,
					}
				},

				Err(err) =>
				{
					error!(error = ?err, "failed to receive a reusable predictions buffer; the channel is closed, stopping the predictor");
					break;
				},
			};

			match self
				.predictions_pool
				.tx()
				.try_send(buffer)
			{
				Ok(true) => (),

				Ok(false) =>
				{
					debug!("dropping a raw prediction because the postprocessor queue is full");
				},

				Err(err) =>
				{
					error!(error = ?err, "failed to forward a raw prediction; the channel is closed, stopping the predictor");
					break;
				},
			}
		}

		info!("predictor stream loop stopped");
	}

	fn process_stream_sample(
		&mut self,
		sample : Sample,
	) -> MlResult<Option<Vec<u8>>>
	{
		if let Some(buffer) = sample.buffer()
			&& let Ok(map) = buffer.map_readable()
		{
			let raw_rgb_data = map.as_slice();

			self.ref_detector
				.update_image_data(raw_rgb_data)
				.map_err(|e| MlError::Reference(e.to_string()))?;

			let tensor_data = unsafe { self._inp.held_data() };

			convert_image::<INP_HEIGHT, INP_WIDTH, INP_CHANNELS>(raw_rgb_data, tensor_data);

			// TEMPORARY DEBUG START: retain selected frames for annotated dumps.
			if DEBUG_SAVE_DETECTIONS
			{
				let frame_idx = DEBUG_SAVE_COUNTER.fetch_add(1, Ordering::Relaxed);

				if frame_idx.is_multiple_of(DEBUG_SAVE_EVERY_N_FRAMES)
				{
					return Ok(Some(raw_rgb_data.to_vec()));
				}
			}
			// TEMPORARY DEBUG END: retain selected frames for annotated dumps.
		}

		Ok(None)
	}
}



pub(crate) struct Postprocessor
{
	predictions_pool : Pool<Prediction>,
	results_tx :       AsyncSender<InferenceResult>,
	_detections :      Vec<Detection<INP_HEIGHT, INP_WIDTH>>,
}

impl Postprocessor
{
	pub(crate) fn new(
		predictions_pool : Pool<Prediction>,
		results_tx : AsyncSender<InferenceResult>,
	) -> Self
	{
		info!("initializing YOLO-World predictions postprocessor");
		Postprocessor {
			predictions_pool,
			results_tx,
			_detections : Vec::with_capacity(30), // initial capacity
		}
	}

	fn validate_detection(
		score : f32,
		obj : &'static Object,
		xc : f32,
		yc : f32,
		width : f32,
		height : f32,
	) -> Option<Detection<INP_HEIGHT, INP_WIDTH>>
	{
		if score < obj.confidence_threshold
		{
			return None;
		}

		if width < obj.min_size
			|| width > obj.max_size
			|| height < obj.min_size
			|| height > obj.max_size
		{
			return None;
		}

		Some(Detection::from_coordinates(obj, score, xc, yc, width, height))
	}

	fn process_raw_detection(
		xc : f32,
		yc : f32,
		width : f32,
		height : f32,
		tracked_score : f32,
		target_score : f32,
		obstacle_score : f32,
	) -> Option<Detection<INP_HEIGHT, INP_WIDTH>>
	{
		let (mut max_score, mut obj) = (tracked_score, &TRACKED);

		if target_score > max_score
		{
			max_score = target_score;
			obj = &TARGET;
		}

		if obstacle_score > max_score
		{
			max_score = obstacle_score;
			obj = &OBSTACLE;
		}

		Self::validate_detection(sigmoid(max_score), obj, xc, yc, width, height)
	}

	fn filter_raw_detections(
		&mut self,
		raw_detections : &[f32],
	)
	{
		if raw_detections.is_empty()
		{
			return;
		}

		let n_detections = OUT_SHAPE.2 as usize;

		let detections_iter = (0 .. n_detections)
			.into_par_iter()
			.filter_map(|idx| {
				Self::process_raw_detection(
					raw_detections[idx],
					raw_detections[n_detections + idx],
					raw_detections[2 * n_detections + idx],
					raw_detections[3 * n_detections + idx],
					raw_detections[4 * n_detections + idx],
					raw_detections[5 * n_detections + idx],
					raw_detections[6 * n_detections + idx],
				)
			});

		self._detections
			.par_extend(detections_iter);
	}

	fn filter_final_detections(&mut self) -> InferenceResult
	{
		let mut result = InferenceResult {
			tracked :   Detection::default(),
			target :    Detection::default(),
			obstacle :  Detection::default(),
			reference : None,
		};

		self._detections
			.iter()
			.for_each(|detection| {
				if detection.object.label == TRACKED.label
					&& detection.confidence > result.tracked.confidence
				{
					result.tracked = *detection;
				}
				else if detection.object.label == OBSTACLE.label
					&& detection.confidence > result.obstacle.confidence
				{
					result.obstacle = *detection;
				}
				else if detection.confidence > result.target.confidence
				{
					result.target = *detection;
				}
			});

		result
	}

	pub(crate) async fn run(&mut self)
	{
		info!("postprocessor loop started");

		while let Ok(mut buffer) = self
			.predictions_pool
			.rx()
			.as_async()
			.recv()
			.await
		{
			self.filter_raw_detections(&buffer.model_detections);

			let reference = buffer.reference_detection;
			let debug_frame = buffer.debug_frame.take();

			if let Err(err) = self
				.predictions_pool
				.tx()
				.try_send(buffer)
			{
				error!(error = ?err, "failed to return predictions buffer to pool");
			}

			if !self._detections.is_empty()
			{
				let mut final_detections = self.filter_final_detections();

				final_detections.reference = reference;

				// TEMPORARY DEBUG START: save annotated detections.
				if let Some(raw_rgb_data) = debug_frame
				{
					Self::spawn_debug_frame_dump(raw_rgb_data, final_detections);
				}
				// TEMPORARY DEBUG END: save annotated detections.

				self._detections.clear();

				if let Err(err) = self
					.results_tx
					.send(final_detections)
					.await
				{
					error!(error = ?err, "failed to forward final detections");
				}
			}
		}

		info!("postprocessor loop stopped");
	}

	// TEMPORARY DEBUG START: save annotated detections.
	fn spawn_debug_frame_dump(
		raw_rgb_data : Vec<u8>,
		detections : InferenceResult,
	)
	{
		std::thread::spawn(move || {
			if let Err(err) = std::fs::create_dir_all(DEBUG_SAVE_DIR)
			{
				error!(error = %err, path = DEBUG_SAVE_DIR, "failed to create debug image directory");
				return;
			}

			let mut image =
				match RgbImage::from_raw(INP_WIDTH as u32, INP_HEIGHT as u32, raw_rgb_data)
				{
					Some(image) => image,
					None =>
					{
						error!("failed to construct debug image from raw RGB frame");
						return;
					},
				};

			for (detection, color) in [
				(detections.tracked(), Rgb([255, 0, 0])),
				(detections.target(), Rgb([0, 0, 255])),
				(detections.obstacle(), Rgb([255, 255, 0])),
			]
			{
				if !detection.is_empty()
				{
					draw_hollow_rect_mut(&mut image, detection.bounding_box.rect(), color);
				}
			}

			if let Some(reference) = detections.reference()
			{
				let center = (reference.center_x, reference.center_y);
				let front = (reference.front_x, reference.front_y);
				let color = Rgb([0, 255, 0]);

				draw_line_segment_mut(&mut image, center, front, color);
				draw_filled_circle_mut(
					&mut image,
					(reference.center_x.round() as i32, reference.center_y.round() as i32),
					4,
					color,
				);
				draw_filled_circle_mut(
					&mut image,
					(reference.front_x.round() as i32, reference.front_y.round() as i32),
					3,
					Rgb([255, 255, 255]),
				);
			}

			let frame_idx = DEBUG_SAVE_COUNTER.load(Ordering::Relaxed);
			let output_path = format!("{DEBUG_SAVE_DIR}/frame_{frame_idx:06}.jpg");

			if let Err(err) = image.save(&output_path)
			{
				error!(error = %err, path = output_path, "failed to save annotated debug frame");
			}
		});
	}
	// TEMPORARY DEBUG END: save annotated detections.
}



#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct InferenceResult
{
	tracked :   Detection<INP_HEIGHT, INP_WIDTH>,
	target :    Detection<INP_HEIGHT, INP_WIDTH>,
	obstacle :  Detection<INP_HEIGHT, INP_WIDTH>,
	reference : Option<Corners>,
}

impl InferenceResult
{
	pub(crate) fn tracked(&self) -> &Detection<INP_HEIGHT, INP_WIDTH> { &self.tracked }

	pub(crate) fn reference(&self) -> &Option<Corners> { &self.reference }

	pub(crate) fn target(&self) -> &Detection<INP_HEIGHT, INP_WIDTH> { &self.target }

	pub(crate) fn obstacle(&self) -> &Detection<INP_HEIGHT, INP_WIDTH> { &self.obstacle }
}



#[cfg(test)]
mod tests
{
	use std::fs;
	use std::io::Write;
	use std::path::{Path, PathBuf};
	use std::time::Instant;

	use image::imageops::FilterType;
	use image::{Rgb, RgbImage};
	use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
	use tracing_test::traced_test;

	use super::*;



	const TEST_IMAGES_DIR : &str = "assets/images";
	const TEST_OUTPUT_SUFFIX : &str = "_detected";
	const TEST_TIMING_FILE : &str = "inference_times.txt";



	fn create_test_postprocessor() -> Postprocessor
	{
		let (_, predictions_pool) = Pool::open(2);
		let (detection_tx, _) = kanal::bounded_async(1);

		Postprocessor::new(predictions_pool, detection_tx)
	}


	fn create_test_marker_detector() -> MarkerDetector<INP_HEIGHT, INP_WIDTH>
	{
		MarkerDetector::init(Default::default()).expect("should initialize marker detector")
	}


	fn get_test_image_paths(images_dir : &Path) -> Vec<PathBuf>
	{
		let mut image_paths = fs::read_dir(images_dir)
			.expect("should read the test images directory")
			.filter_map(|entry| entry.ok().map(|entry| entry.path()))
			.filter(|path| {
				path.extension()
					.and_then(|ext| ext.to_str())
					.is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
					&& path
						.file_stem()
						.and_then(|stem| stem.to_str())
						.is_some_and(|stem| !stem.ends_with(TEST_OUTPUT_SUFFIX))
			})
			.collect::<Vec<_>>();

		image_paths.sort();

		image_paths
	}


	fn load_test_image(image_path : &Path) -> (RgbImage, Vec<u8>)
	{
		let source_image = image::open(image_path)
			.expect("should load source image")
			.to_rgb8();

		let resized_image = image::DynamicImage::ImageRgb8(source_image)
			.resize_exact(INP_WIDTH as u32, INP_HEIGHT as u32, FilterType::Triangle)
			.to_rgb8();

		let raw_rgb_data = resized_image.as_raw().clone();

		(resized_image, raw_rgb_data)
	}


	fn output_image_path(image_path : &Path) -> PathBuf
	{
		let stem = image_path
			.file_stem()
			.and_then(|stem| stem.to_str())
			.expect("test image should have a valid file stem");

		image_path.with_file_name(format!("{stem}{TEST_OUTPUT_SUFFIX}.jpg"))
	}


	fn draw_detections(
		image : &mut RgbImage,
		detections : &InferenceResult,
	)
	{
		[
			(detections.tracked(), Rgb([255, 0, 0])),
			(detections.target(), Rgb([0, 0, 255])),
		]
		.into_iter()
		.filter(|(detection, _)| !detection.is_empty())
		.for_each(|(detection, color)| {
			draw_hollow_rect_mut(image, detection.bounding_box.rect(), color);
		});

		if let Some(reference) = detections.reference()
		{
			let center = (reference.center_x, reference.center_y);
			let front = (reference.front_x, reference.front_y);
			let color = Rgb([0, 255, 0]);

			draw_line_segment_mut(image, center, front, color);
			draw_filled_circle_mut(
				image,
				(reference.center_x.round() as i32, reference.center_y.round() as i32),
				4,
				color,
			);
			draw_filled_circle_mut(
				image,
				(reference.front_x.round() as i32, reference.front_y.round() as i32),
				3,
				Rgb([255, 255, 255]),
			);
		}
	}


	fn timing_output_path(images_dir : &Path) -> PathBuf { images_dir.join(TEST_TIMING_FILE) }


	fn save_timing_report(
		images_dir : &Path,
		image_paths : &[PathBuf],
		inference_times_ms : &[f64],
	)
	{
		let mut file = fs::File::create(timing_output_path(images_dir))
			.expect("should create timing report file");

		let avg_ms = inference_times_ms.iter().sum::<f64>() / inference_times_ms.len() as f64;

		writeln!(file, "average_ms={avg_ms:.3}").expect("should write average inference time");

		image_paths
			.iter()
			.zip(inference_times_ms.iter())
			.for_each(|(image_path, inference_time_ms)| {
				let image_name = image_path
					.file_name()
					.and_then(|name| name.to_str())
					.expect("test image should have a valid file name");

				writeln!(file, "{image_name}={inference_time_ms:.3}")
					.expect("should write per-image inference time");
			});
	}



	#[test]
	#[traced_test]
	#[cfg(target_os = "macos")]
	fn test_yolo_object_detection()
	{
		let (_, rx) = kanal::bounded(1);
		let pool = Pool::open(2);
		let ref_detector = create_test_marker_detector();

		let mut yolo = Predictor::init(rx, ref_detector, pool.1).expect("should initialize");
		let mut postprocessor = create_test_postprocessor();
		let mut ref_detector = create_test_marker_detector();
		let images_dir = Path::new(TEST_IMAGES_DIR);
		let image_paths = get_test_image_paths(images_dir);
		let mut inference_times_ms = Vec::with_capacity(image_paths.len());

		assert!(!image_paths.is_empty(), "should have at least one test image");

		let warm_up_input =
			vec![0_u8; INP_HEIGHT as usize * INP_WIDTH as usize * INP_CHANNELS as usize];
		let tensor_data = unsafe { yolo._inp.held_data() };
		convert_image::<INP_HEIGHT, INP_WIDTH, INP_CHANNELS>(&warm_up_input, tensor_data);
		let _ = yolo
			.engine
			.infer()
			.expect("warm-up inference should succeed");

		image_paths
			.iter()
			.for_each(|image_path| {
				let (mut resized_image, raw_rgb_data) = load_test_image(image_path);

				let tensor_data = unsafe { yolo._inp.held_data() };
				convert_image::<INP_HEIGHT, INP_WIDTH, INP_CHANNELS>(&raw_rgb_data, tensor_data);

				let inference_start = Instant::now();
				let mut outputs = yolo
					.engine
					.infer()
					.expect("inference should succeed");
				inference_times_ms.push(inference_start.elapsed().as_secs_f64() * 1000.0);

				let boxes_value = outputs
					.remove(OUT_KEY)
					.expect("output tensor should be present");

				let boxes_data : &[f32] = boxes_value
					.try_extract_tensor::<f32>()
					.expect("output tensor should be extractable")
					.1;

				postprocessor.filter_raw_detections(boxes_data);
				ref_detector
					.update_image_data(&raw_rgb_data)
					.expect("marker detector image update should succeed");
				let mut detections = postprocessor.filter_final_detections();
				detections.reference = ref_detector
					.detect()
					.expect("marker detection should succeed");
				draw_detections(&mut resized_image, &detections);

				resized_image
					.save(output_image_path(image_path))
					.expect("annotated image should be written");
			});

		save_timing_report(images_dir, &image_paths, &inference_times_ms);
	}
}
