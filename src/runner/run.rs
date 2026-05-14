use gstreamer as gst;
use tokio::time::{Duration, sleep};
use tracing::{debug, error, info};
use uuid::uuid;

use crate::com::{BLECom, Pool};
use crate::control::{ControlParameters, Controller};
use crate::cv::obj_detect::{MarkerDetector, MarkerDetectorParameters};
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
const RESTART_DELAY : Duration = Duration::from_secs(2);



async fn initialize_bluetooth() -> BLECom
{
	info!("connecting bluetooth transport");

	loop
	{
		match BLECom::connect(BLE_DEVICE_NAME, uuid!(BLE_DEVICE_UUID)).await
		{
			Ok(blecom) => return blecom,

			Err(e) =>
			{
				error!(error = %e, "failed to establish Bluetooth connection");

				sleep(Duration::from_secs(1)).await;
			},
		}
	}
}


// noinspection ALL
pub async fn run()
{
	init_logging();

	info!("starting robot control runtime");

	gst::init().expect("failed to initialize GStreamer");

	let mut restart_count = 0usize;

	loop
	{
		restart_count += 1;
		info!(restart_count, "starting runtime attempt");

		// TODO: remove it afterwards
		sleep(Duration::from_secs(2)).await;

		let (tx, rx) = kanal::bounded(FRAMES_POOL_SIZE);

		info!("starting camera stream");
		let stream = match Stream::new(
			STREAM_SINK_NAME,
			DEVICE_IDX,
			StreamParameters::default(),
			tx, // TODO: change that to pool
		)
		{
			Ok(stream) => stream,

			Err(e) =>
			{
				error!(
					error = %e,
					restart_delay_secs = RESTART_DELAY.as_secs(),
					"camera capture pipeline initialization failed, restarting runtime"
				);
				sleep(RESTART_DELAY).await;
				continue;
			},
		};

		debug!("opening raw predictions pool");
		let (pool_a, pool_b) = Pool::open(RAW_PREDICTIONS_POOL_SIZE);

		info!("initializing predictor");
		let marker_detector = match MarkerDetector::init(MarkerDetectorParameters::default())
		{
			Ok(marker_detector) => marker_detector,

			Err(e) =>
			{
				error!(
					error = %e,
					restart_delay_secs = RESTART_DELAY.as_secs(),
					"failed to initialize ArUco marker detector, restarting runtime"
				);
				sleep(RESTART_DELAY).await;
				continue;
			},
		};

		let mut predictor = match Predictor::init(rx, marker_detector, pool_a)
		{
			Ok(predictor) => predictor,

			Err(e) =>
			{
				error!(
					error = %e,
					restart_delay_secs = RESTART_DELAY.as_secs(),
					"YOLO-World model initialization failed, restarting runtime"
				);
				sleep(RESTART_DELAY).await;
				continue;
			},
		};

		debug!("opening the final detections channel");
		let (tx, rx) = kanal::bounded_async(FINAL_DETECTIONS_CHANNEL_CAPACITY);

		info!("initializing postprocessor");
		let mut postprocessor = Postprocessor::new(pool_b, tx);

		let blecom = initialize_bluetooth().await;

		info!("initializing controller");
		let mut controller = Controller::new(ControlParameters::default(), blecom, rx);

		debug!("spawning predictor worker");
		let mut predictor_handle = tokio::task::spawn_blocking(move || {
			info!("predictor worker started");
			predictor.run_stream();
		});

		debug!("spawning postprocessor worker");
		let mut postprocessor_handle = tokio::spawn(async move {
			info!("postprocessor worker started");
			postprocessor.run().await;
		});

		debug!("spawning controller worker");
		let mut controller_handle = tokio::spawn(async move {
			info!("controller worker started");
			controller.run().await
		});

		debug!("spawning camera pipeline worker");
		let mut stream_handle = tokio::task::spawn_blocking(move || {
			info!("camera pipeline worker started");
			stream.start()
		});

		let stopped_component = tokio::select! {
			res = &mut stream_handle => {
				match res
				{
					Ok(Ok(())) => info!("camera pipeline worker stopped"),
					Ok(Err(e)) => error!(error = %e, "camera pipeline worker failed"),
					Err(e) => error!(error = %e, "camera pipeline worker join failed"),
				}
				"camera pipeline"
			},

			res = &mut predictor_handle => {
				match res
				{
					Ok(()) => info!("predictor worker stopped"),
					Err(e) => error!(error = %e, "predictor worker join failed"),
				}
				"predictor"
			},

			res = &mut postprocessor_handle => {
				match res
				{
					Ok(()) => info!("postprocessor worker stopped"),
					Err(e) => error!(error = %e, "postprocessor worker join failed"),
				}
				"postprocessor"
			},

			res = &mut controller_handle => {
				match res
				{
					Ok(Ok(true)) => info!("controller worker stopped"),
					Ok(Ok(false)) => error!("controller worker requested a runtime restart"),
					Ok(Err(e)) => error!(error = %e, "controller worker failed"),
					Err(e) => error!(error = %e, "controller worker join failed"),
				}
				"controller"
			},
		};

		error!(
			component = stopped_component,
			restart_delay_secs = RESTART_DELAY.as_secs(),
			"runtime worker stopped, restarting the runtime"
		);

		if stopped_component != "postprocessor"
		{
			postprocessor_handle.abort();
			let _ = postprocessor_handle.await;
		}

		if stopped_component != "controller"
		{
			controller_handle.abort();
			let _ = controller_handle.await;
		}

		if stopped_component != "predictor"
			&& let Err(e) = predictor_handle.await
		{
			error!(error = %e, "predictor worker join failed during restart cleanup");
		}

		if stopped_component != "camera pipeline"
		{
			match stream_handle.await
			{
				Ok(Ok(())) => info!("camera pipeline worker stopped during restart cleanup"),
				Ok(Err(e)) =>
				{
					error!(error = %e, "camera pipeline worker failed during restart cleanup")
				},
				Err(e) =>
				{
					error!(error = %e, "camera pipeline worker join failed during restart cleanup")
				},
			}
		}

		sleep(RESTART_DELAY).await;
	}
}
