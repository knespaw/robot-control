use rayon::prelude::*;



const IMG_CONV_FACTOR : f32 = 1.0 / 255.0;



/// `img_data` represents a raw RGB image in **HWC** format represented as [`u8`] pixel values.
///
/// Values are converted to [`f32`] and format is changed to **CHW**.
///
/// `target_data` is overwritten with this new data.
pub(crate) fn convert_image<const H: u16, const W: u16, const C: u16>(
	img_data : &[u8],
	target_data : &mut [f32],
)
{
	let c_usize = C as usize;
	let channel_size = H as usize * W as usize;
	let size = channel_size * c_usize;

	assert_eq!(img_data.len(), size);
	assert_eq!(target_data.len(), size);

	target_data
		.par_chunks_exact_mut(channel_size)
		.enumerate()
		.for_each(|(c, channel)| {
			channel
				.iter_mut()
				.enumerate()
				.for_each(|(pos, pixel)| {
					let src_idx = pos * c_usize + c;
					*pixel = img_data[src_idx] as f32 * IMG_CONV_FACTOR;
				})
		});
}
