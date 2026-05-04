use std::f32::consts::PI;

use kanal::AsyncReceiver;

use super::regulator::VelocityRegulator;
use super::tracker::ObjectTracker;
use crate::com::BLECom;
use crate::ml::InferenceResult;



pub(crate) struct ControlParameters
{
	/// Maximum angular velocity of the robot (in rad/s).
	max_angular_velocity :         f32,
	/// Maximum linear velocity of the robot (in mm/s).
	max_linear_velocity :          f32,
	/// Tuning coefficient influencing the turning speed.
	steering_coefficient :         f32,
	/// Tuning coefficient influencing the forward speed.
	forward_coefficient :          f32,
	/// **EMA** value used to smooth positions of detected objects.
	position_smoothing_parameter : f32,
	/// Maximum expected value of an object's position change between consecutive frames. Updates
	/// exceeding this threshold are not taken into account. Defined as a **square** distance.
	position_change_threshold :    f32,
	/// Rotation angle (in radians) between the tracked object and its orientation reference
	/// object.
	reference_rotation :           f32,
	/// Maximum number of consecutive frames that were missing any detection.
	missed_detections_threshold :  usize,
}

impl Default for ControlParameters
{
	fn default() -> Self
	{
		ControlParameters {
			max_angular_velocity :         0.75,
			max_linear_velocity :          150.0,
			steering_coefficient :         2.0, // TODO
			forward_coefficient :          0.5, // TODO
			position_smoothing_parameter : 0.5,
			position_change_threshold :    100.0, // TODO
			reference_rotation :           -0.5 * PI,
			missed_detections_threshold :  10, // TODO
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
		Controller {
			tracker : ObjectTracker::new(
				params.position_change_threshold,
				params.position_smoothing_parameter,
				params.reference_rotation,
			),
			regulator : VelocityRegulator::new(
				params.max_angular_velocity,
				params.steering_coefficient,
				params.max_linear_velocity,
				params.forward_coefficient,
			),
			no_updates_threshold : params.missed_detections_threshold,
			no_updates_count : 0,
			message : String::new(),
			blecom : bluetooth_com,
			detections_rx,
		}
	}

	fn no_updates_check(
		&mut self,
		misses : usize,
	)
	{
		if misses > 0
		{
			self.no_updates_count += misses;

			if self.no_updates_count > self.no_updates_threshold
			{
				todo!()
			}
		}
		else
		{
			self.no_updates_count = 0;
		}
	}

	fn process_detections(
		&mut self,
		detections : &InferenceResult,
	)
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

		if !detections.reference().is_empty()
		{
			self.tracker
				.update_reference(&detections.reference().bounding_box);
		}
		else
		{
			n_missing_detections += 1;
		}

		self.no_updates_check(n_missing_detections);

		let pos_vec = self
			.tracker
			.calculate_corrected_position_vector();

		self.regulator.set_velocities(pos_vec);
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
	}

	pub(crate) async fn run(&mut self)
	{
		while let Ok(new_detections) = self.detections_rx.recv().await
		{
			self.process_detections(&new_detections);

			self.prepare_message();

			self.blecom
				.write(self.message.as_bytes())
				.await
				.unwrap(); // TODO
		}
	}
}
