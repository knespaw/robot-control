use kanal::AsyncReceiver;

use super::regulator::VelocityRegulator;
use super::tracker::ObjectTracker;
use crate::com::BLECom;
use crate::ml::InferenceResult;



struct Controller
{
	tracker :              ObjectTracker,
	regulator :            VelocityRegulator,
	detections_rx :        AsyncReceiver<InferenceResult>,
	no_updates_count :     usize,
	no_updates_threshold : usize,
	blecom :               BLECom,
	message :              String,
}

impl Controller
{
	fn no_updates_check(
		&mut self,
		updated : bool,
	)
	{
		if !updated
		{
			self.no_updates_count += 1;

			if self.no_updates_count > self.no_updates_threshold
			{
				todo!()
			}
		}
		else
		{
			self.no_updates_count = 0;
		}
	}

	fn process_detections(
		&mut self,
		detections : &InferenceResult,
	)
	{
		if let Some(tracked_obj) = detections[0]
		{
			if let Some(ref_obj) = detections[1]
			{
				let pos_vec = self
					.tracker
					.calculate_corrected_position_vector(
						&tracked_obj.bounding_box,
						&ref_obj.bounding_box,
					);

				self.regulator.set_velocities(pos_vec);

				self.no_updates_check(true);
			}
			else
			{
				self.no_updates_check(false);
			}
		}
		else
		{
			self.no_updates_check(false);
		}
	}

	fn prepare_message(&mut self)
	{
		self.message.clear();
		self.message.push('V');
		self.message.push_str(
			&self
				.regulator
				.linear_velocity()
				.to_string(),
		);
		self.message.push('W');
		self.message.push_str(
			&self
				.regulator
				.angular_velocity()
				.to_string(),
		);
		self.message.push('\n');
	}

	async fn run(&mut self) // TODO: doing ops async?
	{
		while let Ok(new_detections) = self.detections_rx.recv().await
		{
			self.process_detections(&new_detections);

			self.prepare_message();

			self.blecom
				.write(self.message.as_bytes())
				.await
				.unwrap(); // TODO
		}
	}
}
