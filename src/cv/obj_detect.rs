use imageproc::rect::Rect;
use opencv::prelude::*;
use opencv::{core, objdetect};

use crate::cv::utils::*;
use crate::ml::{INP_HEIGHT, INP_WIDTH};



const EMPTY_OBJECT : Object = Object {
	label :                "",
	confidence_threshold : 0.0,
	iou_threshold :        0.0,
	max_size :             0.0,
	min_size :             0.0,
};



type Points = core::Vector<core::Point2f>;



#[derive(Copy, Clone, Debug)]
pub(crate) struct Object
{
	pub(crate) label :                &'static str,
	pub(crate) confidence_threshold : f32,
	pub(crate) max_size :             f32,
	pub(crate) min_size :             f32,
	#[allow(dead_code)]
	pub(crate) iou_threshold :        f32,
}



#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct Corners
{
	pub(crate) center_x : f32,
	pub(crate) center_y : f32,
	pub(crate) front_x :  f32,
	pub(crate) front_y :  f32,
}

impl Corners
{
	fn new(points : Points) -> CvResult<Self>
	{
		let top_left = points
			.get(0)
			.map_err(CvError::MarkerDetection)?;
		let top_right = points
			.get(1)
			.map_err(CvError::MarkerDetection)?;
		let bottom_right = points
			.get(2)
			.map_err(CvError::MarkerDetection)?;
		let bottom_left = points
			.get(3)
			.map_err(CvError::MarkerDetection)?;

		let center_x = (top_left.x + top_right.x + bottom_left.x + bottom_right.x) / 4.0;
		let center_y = (top_left.y + top_right.y + bottom_left.y + bottom_right.y) / 4.0;

		let front_x = (top_left.x + top_right.x) / 2.0;
		let front_y = (top_left.y + top_right.y) / 2.0;

		Ok(Corners { center_y, center_x, front_x, front_y })
	}
}



#[derive(Default, Clone, Copy, Debug)]
#[allow(dead_code)]
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

	#[allow(dead_code)]
	pub(crate) fn rect(&self) -> Rect
	{
		Rect::at(self.x1.round() as i32, self.y1.round() as i32)
			.of_size(self.width.round() as u32, self.height.round() as u32)
	}

	#[allow(dead_code)]
	fn area(&self) -> f32 { (self.x2 - self.x1) * (self.y2 - self.y1) }

	#[allow(dead_code)]
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



#[derive(Clone, Copy, Debug)]
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



pub(crate) struct MarkerDetectorParameters
{
	dict_type :             objdetect::PredefinedDictionaryType,
	min_rep_distance :      f32,
	error_correction_rate : f32,
	check_all_borders :     bool,
	img_height :            u16,
	img_width :             u16,
}

impl Default for MarkerDetectorParameters
{
	fn default() -> Self
	{
		MarkerDetectorParameters {
			dict_type :             objdetect::PredefinedDictionaryType::DICT_4X4_50,
			min_rep_distance :      10.0,
			error_correction_rate : 3.0,
			check_all_borders :     true,
			img_height :            INP_HEIGHT,
			img_width :             INP_WIDTH,
		}
	}
}



pub(crate) struct MarkerDetector<const H: u16, const W: u16>
{
	detector :  objdetect::ArucoDetector,
	_corners :  core::Vector<Points>,
	_ids :      core::Vector<i32>,
	_rejected : core::Vector<Points>,
	_img_data : Mat,
}

impl<const H: u16, const W: u16> MarkerDetector<H, W>
{
	pub(crate) fn init(params : MarkerDetectorParameters) -> CvResult<Self>
	{
		let dict =
			objdetect::get_predefined_dictionary(params.dict_type).map_err(CvError::MarkerInit)?;

		let det_params = objdetect::DetectorParameters::default().map_err(CvError::MarkerInit)?;

		let detector = objdetect::ArucoDetector::new(
			&dict,
			&det_params,
			objdetect::RefineParameters::new(
				params.min_rep_distance,
				params.error_correction_rate,
				params.check_all_borders,
			)
			.map_err(CvError::MarkerInit)?,
		)
		.map_err(CvError::MarkerInit)?;

		let _img_data = Mat::new_rows_cols_with_default(
			params.img_height as i32,
			params.img_width as i32,
			core::CV_8UC3,
			core::Scalar::all(0.0),
		)
		.map_err(CvError::MarkerInit)?;

		Ok(MarkerDetector {
			detector,
			_img_data,
			_ids : core::Vector::new(),
			_corners : core::Vector::new(),
			_rejected : core::Vector::new(),
		})
	}

	pub(crate) fn update_image_data(
		&mut self,
		img_data : &[u8],
	) -> CvResult<()>
	{
		let bytes = self
			._img_data
			.data_bytes_mut()
			.map_err(CvError::MarkerUpdate)?;

		bytes.copy_from_slice(img_data);

		Ok(())
	}

	pub(crate) fn detect(&mut self) -> CvResult<Option<Corners>>
	{
		self.detector
			.detect_markers(
				&self._img_data,
				&mut self._corners,
				&mut self._ids,
				&mut self._rejected,
			)
			.map_err(CvError::MarkerDetection)?;

		if self._ids.is_empty()
		{
			return Ok(None);
		}

		// there's only one marker used with id equal to 0
		let marker_corners = self
			._corners
			.get(0)
			.map_err(CvError::MarkerDetection)?;

		let corners = Corners::new(marker_corners)?;

		Ok(Some(corners))
	}
}
