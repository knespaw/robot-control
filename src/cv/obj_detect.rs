use imageproc::rect::Rect;



const EMPTY_OBJECT : Object = Object {
	label :                "",
	confidence_threshold : 0.0,
	iou_threshold :        0.0,
	max_size :             0.0,
	min_size :             0.0,
};



#[derive(Copy, Clone)]
pub(crate) struct Object
{
	pub(crate) label :                &'static str,
	pub(crate) confidence_threshold : f32,
	pub(crate) max_size :             f32,
	pub(crate) min_size :             f32,
	#[allow(dead_code)]
	pub(crate) iou_threshold :        f32,
}



#[derive(Default, Clone, Copy)]
pub(crate) struct BoundingBox<const H: u16, const W: u16>
{
	pub(crate) x1 :     f32,
	pub(crate) x2 :     f32,
	pub(crate) y1 :     f32,
	pub(crate) y2 :     f32,
	pub(crate) xc :     f32,
	pub(crate) yc :     f32,
	pub(crate) height : f32,
	pub(crate) width :  f32,
}

impl<const H: u16, const W: u16> BoundingBox<H, W>
{
	fn new(
		mut xc : f32,
		mut yc : f32,
		mut width : f32,
		mut height : f32,
	) -> Self
	{
		let (w, h) = (W as f32, H as f32);

		xc = xc.clamp(0.0, w);
		yc = yc.clamp(0.0, h);
		width = width.clamp(0.0, w);
		height = height.clamp(0.0, h);

		let x1 = (xc - width / 2.0).clamp(0.0, xc);
		let x2 = (xc + width / 2.0).clamp(xc, w);
		let y1 = (yc - height / 2.0).clamp(0.0, yc);
		let y2 = (yc + height / 2.0).clamp(yc, h);

		BoundingBox { x1, y1, x2, y2, xc, yc, width, height }
	}

	pub(crate) fn rect(&self) -> Rect
	{
		Rect::at(self.x1.round() as i32, self.y1.round() as i32)
			.of_size(self.width.round() as u32, self.height.round() as u32)
	}

	fn area(&self) -> f32 { (self.x2 - self.x1) * (self.y2 - self.y1) }

	pub(crate) fn iou(
		&self,
		other : &Self,
	) -> f32
	{
		if let Some(overlap) = self.rect().intersect(other.rect())
		{
			let overlap_area = (overlap.width() * overlap.height()) as f32;

			let union_area = self.area() + other.area() - overlap_area;

			overlap_area / union_area // overlap_area will always be smaller or equal
		}
		else
		{
			0.0
		}
	}
}



#[derive(Clone, Copy)]
pub(crate) struct Detection<const H: u16, const W: u16>
{
	pub(crate) bounding_box : BoundingBox<H, W>,
	pub(crate) confidence :   f32,
	pub(crate) object :       &'static Object,
}

impl<const H: u16, const W: u16> Detection<H, W>
{
	pub(crate) fn from_coordinates(
		object : &'static Object,
		confidence : f32,
		xc : f32,
		yc : f32,
		width : f32,
		height : f32,
	) -> Self
	{
		let bounding_box = BoundingBox::new(xc, yc, width, height);

		Detection { object, confidence, bounding_box }
	}

	pub(crate) fn is_empty(&self) -> bool { self.object.label.is_empty() }
}

impl<const H: u16, const W: u16> Default for Detection<H, W>
{
	fn default() -> Self
	{
		Detection {
			bounding_box : Default::default(),
			confidence :   0.0,
			object :       &EMPTY_OBJECT,
		}
	}
}
