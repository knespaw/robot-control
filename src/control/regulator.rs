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
	pub(crate) fn linear_velocity(&self) -> f32 { self.linear_velocity }

	pub(crate) fn angular_velocity(&self) -> f32 { self.angular_velocity }

	pub(crate) fn set_velocities(
		&mut self,
		pos_vec : PositionVector,
	)
	{
		self.angular_velocity =
			(self.steering_proportional * pos_vec.1).clamp(-self.max_angular, self.max_angular);

		self.linear_velocity = (self.forward_proportional * pos_vec.0 * pos_vec.1.cos())
			.clamp(-self.max_linear, self.max_linear);
	}
}
