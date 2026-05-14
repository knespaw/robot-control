use std::f32::consts::PI;

use tracing::debug;

use crate::cv::obj_detect::{BoundingBox, Corners};
use crate::ml::{INP_HEIGHT, INP_WIDTH};
use crate::utils::comp::fast_atan2;



type Position = BoundingBox<INP_HEIGHT, INP_WIDTH>;



pub(crate) type PositionVector = (f32, f32);



/// Calculates a distance between point A and B (**squared**), and returns it alongside differences
/// in coordinates.
fn calculate_delta(
	xa : f32,
	ya : f32,
	xb : f32,
	yb : f32,
) -> (f32, f32, f32)
{
	let dx = xa - xb;
	let dy = ya - yb;
	(dx.mul_add(dx, dy * dy), dx, dy)
}



struct PositionSmoother
{
	x :            f32,
	y :            f32,
	_initialized : bool,
}

impl PositionSmoother
{
	/// Updates coordinates of an object by applying the **Exponential Moving Average (EMA)**.
	///
	/// Smoothed coordinates are calculated according to the formula:
	/// `smoothed_val = (1 - alpha) * curr_val + alpha * old_smoothed_val`.
	///
	/// Moreover, `max_delta` determines the maximum possible position change that can happen
	/// within a single update; if the value is exceeded, then no update is done as new coordinates
	/// are considered outliers. It should be given as a **squared** distance between old and
	/// new [`Bounding Box`] **centers**.
	///
	///
	/// [`Bounding Box`]: BoundingBox
	fn update(
		&mut self,
		pos : &Position,
		alpha : f32,
		max_delta : f32,
	)
	{
		if !self._initialized
		{
			self.x = pos.xc;
			self.y = pos.yc;
			self._initialized = true;
		}
		else if calculate_delta(pos.xc, pos.yc, self.x, self.y).0 <= max_delta
		{
			self.x = alpha * pos.xc + (1.0 - alpha) * self.x;
			self.y = alpha * pos.yc + (1.0 - alpha) * self.y;
		}
	}
}



pub(crate) struct ObjectTracker
{
	tracked_pos :         PositionSmoother,
	target_pos :          PositionSmoother,
	reference :           Corners,
	max_pos_change :      f32,
	smoothing_parameter : f32,
}

impl ObjectTracker
{
	pub(crate) fn new(
		max_pos_change : f32,
		smoothing_parameter : f32,
	) -> Self
	{
		ObjectTracker {
			max_pos_change,
			smoothing_parameter,
			tracked_pos : PositionSmoother {
				x :            0.0,
				y :            0.0,
				_initialized : false,
			},
			target_pos : PositionSmoother {
				x :            0.0,
				y :            0.0,
				_initialized : false,
			},
			reference : Corners::default(),
		}
	}

	/// Calculates the distance between both objects' [`Bounding Box`] centers.
	///
	///
	/// [`Bounding Box`]: BoundingBox
	fn calculate_position_vector(
		pos_a : &PositionSmoother,
		pos_b : &PositionSmoother,
	) -> PositionVector
	{
		let (dist, dx, dy) = calculate_delta(pos_a.x, pos_a.y, pos_b.x, pos_b.y);
		let angle = fast_atan2(-dy, dx);

		(dist.sqrt(), angle)
	}

	/// Adjusts the angle which sets the angular velocity of the tracked object (in radians).
	///
	/// It is defined as a difference between `target_angle` and the tracked object's current
	/// heading normalized within
	/// <-[`PI`], [`PI`]> bounds.
	///
	/// If the resulting angle is equal to 0, then the tracked robot perfectly faces the target;
	/// if it is negative, then the object must turn left; otherwise, the object must turn right.
	fn adjust_angle(
		&self,
		target_angle : f32,
	) -> f32
	{
		let forward_heading = fast_atan2(
			-(self.reference.front_y - self.reference.center_y),
			self.reference.front_x - self.reference.center_x,
		);

		let mut angle_error = target_angle - forward_heading;

		// normalizing the angle so that it is within <-180, 180> degrees (to avoid excessive
		// rotations)
		let normalizer = 2.0 * PI;

		while angle_error > PI
		{
			angle_error -= normalizer;
		}

		while angle_error < -PI
		{
			angle_error += normalizer;
		}

		angle_error
	}

	pub(crate) fn update_target(
		&mut self,
		target : &Position,
	)
	{
		self.target_pos
			.update(target, self.smoothing_parameter, self.max_pos_change);
	}

	pub(crate) fn update_tracked(
		&mut self,
		tracked : &Position,
	)
	{
		self.tracked_pos
			.update(tracked, self.smoothing_parameter, self.max_pos_change);
	}

	pub(crate) fn update_reference(
		&mut self,
		reference : &Corners,
	)
	{
		self.reference = *reference;
	}

	pub(crate) fn calculate_corrected_position_vector(&mut self) -> PositionVector
	{
		let (distance, target_angle) =
			Self::calculate_position_vector(&self.target_pos, &self.tracked_pos);

		let corrected_angle = self.adjust_angle(target_angle);

		debug!(
			target_distance = distance,
			corrected_angle, "calculated corrected position vector",
		);

		(distance, corrected_angle)
	}
}
