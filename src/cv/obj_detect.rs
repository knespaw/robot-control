use imageproc::rect::Rect;



#[derive(Default, Clone, Copy)]
pub(crate) struct BoundingBox<const H: usize, const W: usize>
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

impl<const H: usize, const W: usize> BoundingBox<H, W>
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

	fn rect(&self) -> Rect
	{
		Rect::at(self.x1.round() as i32, self.y1.round() as i32)
			.of_size(self.width.round() as u32, self.height.round() as u32)
	}

	fn area(&self) -> f32 { self.width * self.height }

	fn iou(
		&self,
		other : &Self,
	) -> f32
	{
		if let Some(overlap) = self.rect().intersect(other.rect())
		{
			let overlap_area = (overlap.width() * overlap.height()) as f32;

			let union_area = self.area() + other.area() - overlap_area;

			overlap_area / union_area // overlap_area will be always smaller or equal
		}
		else
		{
			0.0
		}
	}
}



#[derive(Default, Clone, Copy)]
pub(crate) struct Detection<const H: usize, const W: usize>
{
	pub(crate) bounding_box : BoundingBox<H, W>,
	pub(crate) confidence :   f32,
	pub(crate) label :        &'static str,
}

impl<const H: usize, const W: usize> Detection<H, W>
{
	pub(crate) fn from_coordinates(
		label : &'static str,
		confidence : f32,
		xc : f32,
		yc : f32,
		width : f32,
		height : f32,
	) -> Self
	{
		let bounding_box = BoundingBox::new(xc, yc, width, height);

		Detection { label, confidence, bounding_box }
	}
}
