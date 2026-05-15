use std::f32::consts::PI;

use tracing::debug;

use crate::cv::obj_detect::{BoundingBox, Corners};
use crate::ml::{INP_HEIGHT, INP_WIDTH};
use crate::utils::comp::fast_atan2;



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



type Position = BoundingBox<INP_HEIGHT, INP_WIDTH>;



pub(crate) type PositionVector = (f32, f32);



struct ArtificialPotentialField
{
	gain :   f32,
	margin : f32,
	vector : (f32, f32),
}

impl ArtificialPotentialField
{
	fn calculate_circumvent_direction(
		tracked : &PositionSmoother,
		obstacle : &PositionSmoother,
		target : &PositionSmoother,
	) -> f32
	{
		let target_dx = target.x - tracked.x;
		let target_dy = target.y - tracked.y;
		let obstacle_dx = obstacle.x - tracked.x;
		let obstacle_dy = obstacle.y - tracked.y;

		// already divided by 2
		if target_dx.mul_add(obstacle_dy, -(target_dy * obstacle_dx)) >= 0.0
		{
			0.5
		}
		else
		{
			-0.5
		}
	}

	/// Calculates the vector of the force of attraction between `target` and `tracked`.
	///
	/// Calculations are done according to:
	/// > 1.1 | *dx*, *dy* - difference in coordinates between `target` and `tracked`
	/// >
	/// > 1.2 | *dist* - distance between `target` and `tracked`
	/// >
	/// > 2.1 | *x = (dx / dist) * gain*, *y = (dy / dist) * gain*
	fn attractive_field(
		&mut self,
		target : &PositionSmoother,
		tracked : &PositionSmoother,
	)
	{
		if !target._initialized || !tracked._initialized
		{
			self.vector = (0.0, 0.0);
			return;
		}

		let (mut dist, dx, dy) = calculate_delta(target.x, target.y, tracked.x, tracked.y);

		if dist == 0.0
		{
			self.vector = (0.0, 0.0);
			return;
		}

		dist = 1.0 / dist.sqrt();

		let x = dx * dist * self.gain;
		let y = dy * dist * self.gain;

		self.vector = (x, y);
	}

	/// Calculates the vector of the force of repulsion between `tracked` and `obstacle`.
	///
	/// Perpendicular force is added (depending on the `tracked` in respect to `obstacle`) to the
	/// vector to make an object flow around (needed to avoid stalling).
	///
	/// Calculations are done according to:
	/// > 1.1. | *dx*, *dy* - difference in coordinates between `tracked` and `obstacle`
	/// >
	/// > 1.2. | *dist* - distance between `tracked` and `obstacle`
	/// >
	/// > 2.1 | *zone_r = max(`b_width`, `b_height`) / 2 + margin*
	/// >
	/// > 2.2 | repulsive force is only applied if: *dist <= zone_r*
	/// >
	/// > 3.1 | *force_mag = gain * (1 / dist - 1 / zone_r) ^ 2*
	/// >
	/// > 3.2 | *x = (dx / dist) * force_mag*, *y = (dy / dist) * force_mag*
	/// >
	/// > 3.3 | *dodge = sign(cross(target_vec, obstacle_vec))*
	/// >
	/// > 3.4 | *x += (-dy / dist) * force_mag * dodge / 2*,
	/// *y += (dx / dist) * force_mag * dodge / 2*
	fn repulsive_field(
		&mut self,
		tracked : &PositionSmoother,
		obstacle : &PositionSmoother,
		target : &PositionSmoother,
		b_width : f32,
		b_height : f32,
	)
	{
		if !tracked._initialized
			|| !obstacle._initialized
			|| !target._initialized
			|| b_width <= 0.0
			|| b_height <= 0.0
		{
			self.vector = (0.0, 0.0);
			return;
		}

		let (mut dist, dx, dy) = calculate_delta(tracked.x, tracked.y, obstacle.x, obstacle.y);

		if dist == 0.0
		{
			self.vector = (0.0, 0.0);
			return;
		}

		dist = 1.0 / dist.sqrt();

		// difference between inversed distance and zone radius
		let dist_zone_r_inv_diff = dist
			- 1.0
				/ b_width
					.max(b_height)
					.mul_add(0.5, self.margin);

		// inverse
		if dist_zone_r_inv_diff >= 0.0
		{
			// multiplied by distance inverse
			let force_mag = dist * dist_zone_r_inv_diff * dist_zone_r_inv_diff * self.gain;

			let x_f = dx * force_mag;
			let y_f = dy * force_mag;

			let dodge_dir = Self::calculate_circumvent_direction(tracked, obstacle, target);

			let x = x_f - y_f * dodge_dir;
			let y = y_f + x_f * dodge_dir;

			self.vector = (x, y);
		}
		else
		{
			self.vector = (0.0, 0.0);
		}
	}
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
	obstacle_pos :        PositionSmoother,
	reference :           Corners,
	max_pos_change :      f32,
	smoothing_parameter : f32,
	attractive_apf :      ArtificialPotentialField,
	repulsive_apf :       ArtificialPotentialField,
}

impl ObjectTracker
{
	pub(crate) fn new(
		max_pos_change : f32,
		smoothing_parameter : f32,
		obstacle_field_gain : f32,
		obstacle_field_margin : f32,
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
			obstacle_pos : PositionSmoother {
				x :            0.0,
				y :            0.0,
				_initialized : false,
			},
			reference : Corners::default(),
			repulsive_apf : ArtificialPotentialField {
				gain :   obstacle_field_gain,
				margin : obstacle_field_margin,
				vector : (0.0, 0.0),
			},
			attractive_apf : ArtificialPotentialField {
				gain :   1.0, // TODO,
				margin : 0.0,
				vector : (0.0, 0.0),
			},
		}
	}

	/// Adjusts the angle which sets the angular velocity of the tracked object (in radians).
	///
	/// It is defined as a difference between the desired heading angle (calculated using
	/// the **APF algorithm**) and the current heading (calculated based on the reference object
	/// position).
	///
	/// The angle is normalized within <-[`PI`], [`PI`]> bounds.
	///
	/// If the resulting angle is equal to 0, then the tracked robot perfectly faces the target;
	/// if it is negative, then the object must turn left; otherwise, the object must turn right.
	fn adjust_angle(&self) -> f32
	{
		//e
		let desired_heading = fast_atan2(
			-(self.attractive_apf.vector.1 + self.repulsive_apf.vector.1),
			self.attractive_apf.vector.0 + self.repulsive_apf.vector.0,
		);

		let current_heading = fast_atan2(
			-(self.reference.front_y - self.reference.center_y),
			self.reference.front_x - self.reference.center_x,
		);

		let mut angle_error = desired_heading - current_heading;

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

	pub(crate) fn update_obstacle(
		&mut self,
		obstacle : &Position,
	)
	{
		self.obstacle_pos
			.update(obstacle, self.smoothing_parameter, self.max_pos_change);
	}

	pub(crate) fn update_reference(
		&mut self,
		reference : &Corners,
	)
	{
		self.reference = *reference;
	}

	pub(crate) fn calculate_apfs(
		&mut self,
		obstacle_width : f32,
		obstacle_height : f32,
	)
	{
		self.attractive_apf
			.attractive_field(&self.target_pos, &self.tracked_pos);

		self.repulsive_apf.repulsive_field(
			&self.tracked_pos,
			&self.obstacle_pos,
			&self.target_pos,
			obstacle_width,
			obstacle_height,
		);
	}

	fn reduce_distance(&mut self) -> f32
	{
		let attractive_force = self
			.attractive_apf
			.vector
			.0
			.hypot(self.attractive_apf.vector.1);

		let repulsive_force = self
			.repulsive_apf
			.vector
			.0
			.hypot(self.repulsive_apf.vector.1);

		if repulsive_force > 0.0
		{
			(1.0 / (1.0 + 3.0 * repulsive_force / attractive_force.max(1e-3))).clamp(0.15, 1.0)
		}
		else
		{
			1.0
		}
	}

	pub(crate) fn calculate_position_vector(&mut self) -> PositionVector
	{
		let (mut target_dist, ..) = calculate_delta(
			self.target_pos.x,
			self.target_pos.y,
			self.tracked_pos.x,
			self.tracked_pos.y,
		);
		target_dist = target_dist.sqrt();

		target_dist *= self.reduce_distance();

		let angle = self.adjust_angle();

		debug!(
			target_distance = target_dist,
			corrected_angle = angle,
			"calculated corrected position vector",
		);

		(target_dist, angle)
	}
}
