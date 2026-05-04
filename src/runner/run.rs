use uuid::uuid;

use crate::com::{BLECom, Pool};
use crate::control::{ControlParameters, Controller};
use crate::cv::{Stream, StreamParameters};
use crate::ml::{Postprocessor, Predictor};



const BLE_DEVICE_NAME : &str = "";
const BLE_DEVICE_UUID : &str = "00000000-0000-0000-0000-ffff00000000";


const STREAM_SINK_NAME : &str = "camera-feed";
const DEVICE_IDX : u8 = 1; // or 2
const FRAMES_POOL_SIZE : usize = 10;


const RAW_PREDICTIONS_POOL_SIZE : usize = 10;
const FINAL_DETECTIONS_CHANNEL_CAPACITY : usize = 10;



pub async fn run()
{
	let (tx, rx) = kanal::bounded(FRAMES_POOL_SIZE);

	let stream = Stream::new(
		STREAM_SINK_NAME,
		DEVICE_IDX,
		StreamParameters::default(),
		tx, // TODO: change that to pool
	)
	.expect("camera capture pipeline start failed");

	let (pool_a, pool_b) = Pool::open(RAW_PREDICTIONS_POOL_SIZE);

	let mut predictor =
		Predictor::init(rx, pool_a).expect("YOLO-World model initialization failed");

	let (tx, rx) = kanal::bounded_async(FINAL_DETECTIONS_CHANNEL_CAPACITY);

	let mut postprocessor = Postprocessor::new(pool_b, tx);

	let blecom = BLECom::connect(BLE_DEVICE_NAME, uuid!(BLE_DEVICE_UUID))
		.await
		.expect("failed to establish Bluetooth connection");

	let mut controller = Controller::new(ControlParameters::default(), blecom, rx);



	tokio::task::spawn_blocking(move || {
		predictor.run_stream();
	});

	tokio::task::spawn_blocking(async move || {
		postprocessor.run().await;
	});

	tokio::task::spawn_blocking(async move || {
		controller.run().await;
	});

	stream
		.start()
		.expect("camera capture failed") // TODO
}
