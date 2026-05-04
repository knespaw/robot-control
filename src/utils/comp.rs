// "Magic" coefficients which are smaller than regular Taylor coefficients to pool the curve down.
const TAYLOR_SERIES_COEFFICIENT_3 : f32 = 0.32762276;
const TAYLOR_SERIES_COEFFICIENT_5 : f32 = 0.15931422;
const TAYLOR_SERIES_COEFFICIENT_7 : f32 = -0.04649647;



/// Fast implementation of the four quadrant arctangent computation for [`f32`] numbers
/// (in radians).
pub(crate) fn fast_atan2(
	y : f32,
	x : f32,
) -> f32
{
	let (abs_y, abs_x) = (y.abs(), x.abs());

	let (min, max) = if abs_x > abs_y { (abs_y, abs_x) } else { (abs_x, abs_y) };

	let r = min / max;
	let r2 = r * r;

	let mut angle = ((TAYLOR_SERIES_COEFFICIENT_7 * r2 + TAYLOR_SERIES_COEFFICIENT_5) * r2
		- TAYLOR_SERIES_COEFFICIENT_3)
		* r2 * r
		+ r;

	if abs_y > abs_x
	{
		angle = std::f32::consts::FRAC_PI_2 - angle;
	}

	if x < 0.0
	{
		angle = std::f32::consts::PI - angle;
	}

	if y < 0.0
	{
		angle = -angle;
	}

	angle
}
