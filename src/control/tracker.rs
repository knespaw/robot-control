use std::f32::consts::PI;

use crate::cv::BoundingBox;
use crate::ml::{INP_HEIGHT, INP_WIDTH};
use crate::utils::comp::fast_atan2;



type Object = BoundingBox<INP_HEIGHT, INP_WIDTH>;


pub(crate) type PositionVector = (f32, f32);



pub(crate) struct ObjectTracker
{
	target :       Object,
	ref_rotation : f32, // -pi / 2 (-90 degrees - to the left - since it's on the left)
}

impl ObjectTracker
{
	/// Calculates the distance (squared) between the target and the tracked object's
	/// [`Bounding Box`] centers.
	///
	/// Square root is not calculated since it is expensive and not needed for comparisons only.
	///
	/// Distance is calculated using *Fused Multiply-Add (FMA)* as it can be done in a single
	/// clock cycle.
	///
	///
	/// [`Bounding Box`]: BoundingBox
	fn calculate_position_vector(
		point_a : &Object,
		point_b : &Object,
	) -> PositionVector
	{
		let dx = point_a.xc - point_b.xc;
		let dy = point_a.yc - point_b.yc;

		let dist = dx.mul_add(dx, dy * dy);
		let angle = fast_atan2(dy, dx);

		(dist, angle)
	}

	/// Adjusts the angle which sets the angular velocity of the tracked object (in radians).
	///
	/// It is defined as a difference between `target_angle` and `ref_angle` normalized within
	/// <-[`PI`], [`PI`]> bounds.
	///
	/// If the resulting angle is equal to 0, then the tracked robot perfectly faces the target;
	/// if it is negative, then the object must turn left; otherwise, the object must turn right.
	fn adjust_angle(
		target_angle : f32,
		ref_angle : f32,
		ref_rotation : f32,
	) -> f32
	{
		let forward_heading = ref_angle + ref_rotation;

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

	pub(crate) fn calculate_corrected_position_vector(
		&self,
		tracked : &Object,
		reference : &Object,
	) -> PositionVector
	{
		let (target_dist, target_angle) = Self::calculate_position_vector(&self.target, tracked);

		let (_, ref_angle) = Self::calculate_position_vector(reference, tracked);

		let corrected_angle = Self::adjust_angle(target_angle, ref_angle, self.ref_rotation);

		(target_dist, corrected_angle)
	}
}
