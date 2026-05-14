use std::f32::consts::FRAC_PI_2;

use tracing::debug;

use super::tracker::PositionVector;



pub(crate) struct VelocityRegulator
{
	angular_velocity :      f32,
	max_angular :           f32,
	steering_proportional : f32,
	linear_velocity :       f32,
	max_linear :            f32,
	forward_proportional :  f32,
}

impl VelocityRegulator
{
	pub(crate) fn new(
		max_angular : f32,
		steering_proportional : f32,
		max_linear : f32,
		forward_proportional : f32,
	) -> Self
	{
		VelocityRegulator {
			max_angular,
			steering_proportional,
			max_linear,
			forward_proportional,
			angular_velocity : 0.0,
			linear_velocity : 0.0,
		}
	}

	pub(crate) fn linear_velocity(&self) -> f32 { self.linear_velocity }

	pub(crate) fn angular_velocity(&self) -> f32 { self.angular_velocity }

	pub(crate) fn set_velocities(
		&mut self,
		pos_vec : PositionVector,
	)
	{
		self.angular_velocity =
			(self.steering_proportional * pos_vec.1).clamp(-self.max_angular, self.max_angular);

		// if the absolute angle is greater than 90 degrees, then the robot is facing wrong
		// direction, and must turn in place
		self.linear_velocity = if pos_vec.1.abs() < FRAC_PI_2
		{
			(self.forward_proportional * pos_vec.0 * pos_vec.1.cos())
				.clamp(-self.max_linear, self.max_linear)
		}
		else
		{
			0.0
		};

		debug!(
			distance = pos_vec.0,
			angle = pos_vec.1,
			linear_velocity = self.linear_velocity,
			angular_velocity = self.angular_velocity,
			"updated control velocities"
		);
	}
}
