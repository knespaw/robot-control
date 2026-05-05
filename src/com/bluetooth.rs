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
use tracing::{error, info};
use uuid::Uuid;

use super::utils::{ComError, ComResult};



pub(crate) struct BLECom
{
	_name :          &'static str,
	_write_uuid :    Uuid,
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
		info!(device_name = name, write_uuid = %write_uuid, "starting bluetooth connection");
		let device = Self::find_device(name).await?;

		let (device, characteristic) = Self::connect_device(device, write_uuid).await?;

		info!(device_name = name, "bluetooth connection established");

		Ok(BLECom {
			_name : name,
			_write_uuid : write_uuid,
			device,
			characteristic,
		})
	}

	async fn find_device(name : &'static str) -> ComResult<Peripheral>
	{
		let manager = Manager::new().await.map_err(|e| {
			error!(device_name = name, error = %e, "failed to create bluetooth manager");
			ComError::BLEStart(e.to_string())
		})?;

		let adapters = manager.adapters().await.map_err(|e| {
			error!(device_name = name, error = %e, "failed to enumerate bluetooth adapters");
			ComError::BLEStart(e.to_string())
		})?;

		let central = adapters
			.into_iter()
			.next()
			.ok_or_else(|| {
				error!(device_name = name, "no bluetooth adapters found");
				ComError::BLEStart("no Bluetooth adapters found".into())
			})?;

		central
			.start_scan(ScanFilter::default())
			.await
			.map_err(|e| {
				error!(device_name = name, error = %e, "failed to start bluetooth scan");
				ComError::BLEStart(e.to_string())
			})?;

		// waiting for the adapter to find devices
		sleep(Duration::from_secs(3)).await;

		let peripherals = central
			.peripherals()
			.await
			.map_err(|e| {
				error!(device_name = name, error = %e, "failed to query peripherals");
				ComError::BLEStart(e.to_string())
			})?;

		let mut device = None;

		for peripheral in peripherals
		{
			if let Some(properties) = peripheral
				.properties()
				.await
				.map_err(|e| {
					error!(device_name = name, error = %e, "failed to query peripheral properties");
					ComError::BLEStart(e.to_string())
				})?
			{
				let local_name = properties
					.local_name
					.unwrap_or_default();

				if local_name == name
				// TODO: will that work? maybe check by address
				{
					info!(device_name = name, "matching bluetooth device found");
					device = Some(peripheral);
					break;
				}
			}
		}

		device.ok_or_else(|| {
			error!(device_name = name, "failed to find bluetooth device");
			ComError::BLEStart(format!("failed to find device `{}`", name))
		})
	}

	async fn connect_device(
		device : Peripheral,
		write_uuid : Uuid,
	) -> ComResult<(Peripheral, Characteristic)>
	{
		info!(write_uuid = %write_uuid, "connecting to bluetooth device");

		device.connect().await.map_err(|e| {
			error!(write_uuid = %write_uuid, error = %e, "failed to connect bluetooth device");
			ComError::BLEStart(e.to_string())
		})?;

		device
			.discover_services()
			.await
			.map_err(|e| {
				error!(write_uuid = %write_uuid, error = %e, "failed to discover bluetooth services");
				ComError::BLEStart(e.to_string())
			})?;

		let characteristic = device
			.characteristics()
			.into_iter()
			.find(|c| c.uuid == write_uuid)
			.ok_or_else(|| {
				error!(write_uuid = %write_uuid, "failed to find bluetooth characteristic");
				ComError::BLEStart(format!("could not find UUID `{}`", write_uuid))
			})?;

		info!(write_uuid = %write_uuid, "bluetooth characteristic ready");

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
			.map_err(|e| {
				error!(payload_len = payload.len(), error = %e, "bluetooth write failed");
				ComError::BLEWrite(e)
			})
	}
}



#[cfg(test)]
mod tests
{
	use tracing_test::traced_test;
	use uuid::uuid;

	use super::*;



	#[tokio::test]
	#[traced_test]
	async fn test_bluetooth_communication()
	{
		let device = "Makeblock_LE001b1062b3bf";
		const UUID : &str = "0000ffe3-0000-1000-8000-00805f9b34fb";

		let blecom = BLECom::connect(device, uuid!(UUID))
			.await
			.expect("should connect");

		let msg = "V70.0W0.0\n";

		for _ in 0 .. 100
		{
			sleep(Duration::from_millis(50)).await;

			blecom
				.write(msg.as_bytes())
				.await
				.expect("should write payload");
		}
	}
}
