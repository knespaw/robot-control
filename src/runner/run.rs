use gstreamer as gst;
use tracing::{debug, error, info};
use uuid::uuid;

use crate::com::{BLECom, Pool};
use crate::control::{ControlParameters, Controller};
use crate::cv::{Stream, StreamParameters};
use crate::ml::{Postprocessor, Predictor};
use crate::utils::logging::init_logging;



const BLE_DEVICE_NAME : &str = "Makeblock_LE001b1062b3bf";
const BLE_DEVICE_UUID : &str = "0000ffe3-0000-1000-8000-00805f9b34fb";


const STREAM_SINK_NAME : &str = "camera-feed";
const DEVICE_IDX : u8 = 1; // or 2
const FRAMES_POOL_SIZE : usize = 10;


const RAW_PREDICTIONS_POOL_SIZE : usize = 10;
const FINAL_DETECTIONS_CHANNEL_CAPACITY : usize = 10;



pub async fn run()
{
	init_logging();

	info!("starting robot control runtime");

	gst::init().expect("failed to initialize GStreamer");

	let (tx, rx) = kanal::bounded(FRAMES_POOL_SIZE);

	info!("starting camera stream");
	let stream = Stream::new(
		STREAM_SINK_NAME,
		DEVICE_IDX,
		StreamParameters::default(),
		tx, // TODO: change that to pool
	)
	.expect("camera capture pipeline start failed");

	debug!("opening raw predictions pool");
	let (pool_a, pool_b) = Pool::open(RAW_PREDICTIONS_POOL_SIZE);

	info!("initializing predictor");
	let mut predictor =
		Predictor::init(rx, pool_a).expect("YOLO-World model initialization failed");

	debug!("opening the final detections channel");
	let (tx, rx) = kanal::bounded_async(FINAL_DETECTIONS_CHANNEL_CAPACITY);

	info!("initializing postprocessor");
	let mut postprocessor = Postprocessor::new(pool_b, tx);

	info!("connecting bluetooth transport");
	let blecom = BLECom::connect(BLE_DEVICE_NAME, uuid!(BLE_DEVICE_UUID))
		.await
		.expect("failed to establish Bluetooth connection");

	info!("initializing controller");
	let mut controller = Controller::new(ControlParameters::default(), blecom, rx);



	debug!("spawning predictor worker");
	tokio::task::spawn_blocking(move || {
		info!("predictor worker started");
		predictor.run_stream();
	});

	debug!("spawning postprocessor worker");
	tokio::task::spawn_blocking(async move || {
		info!("postprocessor worker started");
		postprocessor.run().await;
	});

	debug!("spawning controller worker");
	tokio::task::spawn_blocking(async move || {
		info!("controller worker started");
		controller.run().await;
	});

	info!("starting camera pipeline loop");
	stream
		.start()
		.map_err(|e| {
			error!(error = %e, "camera capture failed");
			e
		})
		.expect("camera capture failed") // TODO
}
