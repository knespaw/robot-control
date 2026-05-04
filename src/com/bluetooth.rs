use btleplug::api::{
	Central,
	Characteristic,
	Manager as _,
	Peripheral as _,
	ScanFilter,
	WriteType,
};
use btleplug::platform::{Manager, Peripheral};
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use super::utils::{ComError, ComResult};



pub(crate) struct BLECom
{
	name :           &'static str,
	write_uuid :     Uuid,
	device :         Peripheral,
	characteristic : Characteristic,
}

impl BLECom
{
	pub(crate) async fn connect(
		name : &'static str,
		write_uuid : Uuid,
	) -> ComResult<Self>
	{
		let device = Self::find_device(name).await?;

		let (device, characteristic) = Self::connect_device(device, write_uuid).await?;

		Ok(BLECom { name, write_uuid, device, characteristic })
	}

	async fn find_device(name : &'static str) -> ComResult<Peripheral>
	{
		let manager = Manager::new()
			.await
			.map_err(|e| ComError::BLEStart(e.to_string()))?;

		let adapters = manager
			.adapters()
			.await
			.map_err(|e| ComError::BLEStart(e.to_string()))?;

		let central = adapters
			.into_iter()
			.next()
			.ok_or(ComError::BLEStart("no Bluetooth adapters found".into()))?;

		central
			.start_scan(ScanFilter::default())
			.await
			.map_err(|e| ComError::BLEStart(e.to_string()))?;

		// waiting for the adapter to find devices
		sleep(Duration::from_secs(3)).await;

		let peripherals = central
			.peripherals()
			.await
			.map_err(|e| ComError::BLEStart(e.to_string()))?;

		let mut device = None;

		for peripheral in peripherals
		{
			if let Some(properties) = peripheral
				.properties()
				.await
				.map_err(|e| ComError::BLEStart(e.to_string()))?
			{
				let local_name = properties
					.local_name
					.unwrap_or_default();

				if local_name == name
				// TODO: will that work? maybe check by address
				{
					device = Some(peripheral);
					break;
				}
			}
		}

		device.ok_or(ComError::BLEStart(format!("failed to find device `{}`", name)))
	}

	async fn connect_device(
		device : Peripheral,
		write_uuid : Uuid,
	) -> ComResult<(Peripheral, Characteristic)>
	{
		device
			.connect()
			.await
			.map_err(|e| ComError::BLEStart(e.to_string()))?;

		device
			.discover_services()
			.await
			.map_err(|e| ComError::BLEStart(e.to_string()))?;

		let characteristic = device
			.characteristics()
			.into_iter()
			.find(|c| c.uuid == write_uuid)
			.ok_or(ComError::BLEStart(format!("could not find UUID `{}`", write_uuid)))?;

		Ok((device, characteristic))
	}

	pub(crate) async fn write(
		&self,
		payload : &[u8],
	) -> ComResult<()>
	{
		self.device
			.write(&self.characteristic, payload, WriteType::WithoutResponse)
			.await
			.map_err(ComError::BLEWrite)
	}
}
