use rayon::prelude::*;



const IMG_CONV_FACTOR : f32 = 1.0 / 255.0;



/// `img_data` represents a raw RGB image in **HWC** format represented as [`u8`] pixel values.
///
/// Values are converted to [`f32`] and format is changed to **CHW**.
///
/// `target_data` is overwritten with this new data.
pub(crate) fn convert_image<const H: usize, const W: usize, const C: usize>(
	img_data : &[u8],
	target_data : &mut [f32],
)
{
	let channel_size = H * W;
	let size = W * H * C;

	debug_assert_eq!(img_data.len(), size);
	debug_assert_eq!(target_data.len(), size);

	target_data
		.par_chunks_exact_mut(channel_size)
		.enumerate()
		.for_each(|(c, channel)| {
			channel
				.iter_mut()
				.enumerate()
				.for_each(|(pos, pixel)| {
					let src_idx = pos * C + c;
					*pixel = img_data[src_idx] as f32 * IMG_CONV_FACTOR;
				})
		});
}
