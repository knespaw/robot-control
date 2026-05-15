use kanal::AsyncReceiver;
use tracing::{debug, error, info, warn};

use super::regulator::VelocityRegulator;
use super::tracker::ObjectTracker;
use crate::com::{BLECom, ComResult};
use crate::ml::InferenceResult;



pub(crate) struct ControlParameters
{
	/// Maximum angular velocity of the robot (in rad/s).
	max_angular_velocity :           f32,
	/// Maximum linear velocity of the robot (in mm/s).
	max_linear_velocity :            f32,
	/// Tuning coefficient influencing the turning speed.
	steering_coefficient :           f32,
	/// Tuning coefficient influencing the forward speed.
	forward_coefficient :            f32,
	/// **EMA** value used to smooth positions of detected objects.
	position_smoothing_parameter :   f32,
	/// Maximum expected value of an object's position change between consecutive frames. Updates
	/// exceeding this threshold are not taken into account. Defined as a **square** distance.
	position_change_threshold :      f32,
	/// Maximum number of consecutive frames that were missing any detection.
	missed_detections_threshold :    usize,
	/// Maximum number of consecutive messages that were failed to be written.
	missed_writes_threshold :        usize,
	obstacle_avoidance_force_gain :  f32,
	obstacle_avoidance_zone_margin : f32,
}

impl Default for ControlParameters
{
	fn default() -> Self
	{
		ControlParameters {
			max_angular_velocity :           0.75,
			max_linear_velocity :            60.0,
			steering_coefficient :           1.0,
			forward_coefficient :            0.5,
			position_smoothing_parameter :   0.5,
			position_change_threshold :      1000.0,
			missed_detections_threshold :    30,
			missed_writes_threshold :        5,
			obstacle_avoidance_force_gain :  2000.0,
			obstacle_avoidance_zone_margin : 300.0,
		}
	}
}



pub(crate) struct Controller
{
	tracker :              ObjectTracker,
	regulator :            VelocityRegulator,
	detections_rx :        AsyncReceiver<InferenceResult>,
	no_updates_count :     usize,
	no_updates_threshold : usize,
	no_writes_count :      usize,
	no_writes_threshold :  usize,
	blecom :               BLECom,
	message :              String,
}

impl Controller
{
	pub(crate) fn new(
		params : ControlParameters,
		bluetooth_com : BLECom,
		detections_rx : AsyncReceiver<InferenceResult>,
	) -> Self
	{
		info!("initializing controller");
		Controller {
			tracker : ObjectTracker::new(
				params.position_change_threshold,
				params.position_smoothing_parameter,
				params.obstacle_avoidance_force_gain,
				params.obstacle_avoidance_zone_margin,
			),
			regulator : VelocityRegulator::new(
				params.max_angular_velocity,
				params.steering_coefficient,
				params.max_linear_velocity,
				params.forward_coefficient,
			),
			no_updates_threshold : params.missed_detections_threshold,
			no_updates_count : 0,
			no_writes_threshold : params.missed_writes_threshold,
			no_writes_count : 0,
			message : String::new(),
			blecom : bluetooth_com,
			detections_rx,
		}
	}

	fn no_updates_check(
		&mut self,
		misses : usize,
	) -> bool
	{
		if misses > 0
		{
			self.no_updates_count += misses;
			warn!(
				misses,
				no_updates_count = self.no_updates_count,
				no_updates_threshold = self.no_updates_threshold,
				"missing detections recorded"
			);

			if self.no_updates_count > self.no_updates_threshold
			{
				error!(
					no_updates_count = self.no_updates_count,
					no_updates_threshold = self.no_updates_threshold,
					"missing detections threshold exceeded"
				);

				return false;
			}
		}
		else
		{
			self.no_updates_count = 0;
		}

		true
	}

	fn update_tracker(
		&mut self,
		detections : &InferenceResult,
	) -> usize
	{
		let mut n_missing_detections = 0;

		if !detections.tracked().is_empty()
		{
			self.tracker
				.update_tracked(&detections.tracked().bounding_box);
		}
		else
		{
			n_missing_detections += 1;
		}

		if !detections.target().is_empty()
		{
			self.tracker
				.update_target(&detections.target().bounding_box);
		}
		else
		{
			n_missing_detections += 1;
		}

		if !detections.obstacle().is_empty()
		{
			self.tracker
				.update_obstacle(&detections.obstacle().bounding_box);
		}
		else
		{
			n_missing_detections += 1;
		}

		if let Some(reference) = detections.reference()
		{
			self.tracker.update_reference(reference);
		}
		else
		{
			n_missing_detections += 1;
		}

		n_missing_detections
	}

	fn process_detections(
		&mut self,
		detections : &InferenceResult,
	) -> bool
	{
		// empty frames
		if detections.target().is_empty()
			&& detections.tracked().is_empty() & detections.reference().is_none()
		{
			return true;
		}

		let misses = self.update_tracker(detections);

		if !self.no_updates_check(misses)
		{
			return false;
		}

		self.tracker.calculate_apfs(
			detections.obstacle().bounding_box.width,
			detections
				.obstacle()
				.bounding_box
				.height,
		);

		let pos_vec = self.tracker.calculate_position_vector();

		self.regulator.set_velocities(pos_vec);

		true
	}

	fn prepare_message(&mut self)
	{
		self.message.clear();
		self.message.push('V');
		self.message.push_str(
			&self
				.regulator
				.linear_velocity()
				.to_string(),
		);
		self.message.push('W');
		self.message.push_str(
			&self
				.regulator
				.angular_velocity()
				.to_string(),
		);
		self.message.push('\n');

		debug!(payload = self.message.as_str(), "prepared control message",);
	}

	pub(crate) async fn run(&mut self) -> ComResult<bool>
	{
		info!("controller loop started");

		while let Ok(new_detections) = self.detections_rx.recv().await
		{
			if !self.process_detections(&new_detections)
			{
				return Ok(false);
			}

			self.prepare_message();

			match self
				.blecom
				.write(self.message.as_bytes())
				.await
			{
				Ok(_) => self.no_writes_count = 0,

				Err(e) =>
				{
					error!(error = %e, "failed to write control message");

					self.no_writes_count += 1;

					if self.no_writes_count > self.no_writes_threshold
					{
						error!(
							no_writes_count = self.no_writes_count,
							no_writes_threshold = self.no_writes_threshold,
							"failed message writes threshold exceeded",
						);

						// the device is then disconnected
						return Err(e);
					}
				},
			}
		}

		info!("controller loop stopped");
		Ok(true)
	}
}
