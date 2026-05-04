use std::io::ErrorKind::InvalidFilename;
use std::path::PathBuf;



pub(crate) fn get_path(
	dir : &str,
	filename : &str,
) -> std::io::Result<String>
{
	let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

	path.push(dir);

	if !filename.is_empty()
	{
		path.push(filename);
	}

	match path.try_exists()
	{
		Ok(true) =>
		{
			Ok(path
				.to_str()
				.expect("should not happen")
				.to_string())
		},

		_ =>
		{
			Err(std::io::Error::new(
				InvalidFilename,
				format!("either file {} or directory {} don't exist", filename, dir),
			))
		},
	}
}
