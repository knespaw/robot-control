use kanal::{Receiver, Sender};



pub(crate) struct Pool<T>
where
	T : Default,
{
	tx : Sender<T>,
	rx : Receiver<T>,
}

impl<T> Pool<T>
where
	T : Default,
{
	pub(crate) fn open(total_pool_size : usize) -> (Self, Self)
	{
		let pool_size = total_pool_size / 2;

		let (mut tx_a, rx_a) = kanal::bounded(pool_size);
		let (mut tx_b, rx_b) = kanal::bounded(pool_size);

		tx_a = Self::prefill(tx_a, pool_size);
		tx_b = Self::prefill(tx_b, pool_size);

		(Pool { tx : tx_a, rx : rx_b }, Pool { tx : tx_b, rx : rx_a })
	}

	fn prefill(
		tx : Sender<T>,
		pool_size : usize,
	) -> Sender<T>
	{
		(0 .. pool_size).for_each(|_| {
			tx.send(T::default()).unwrap(); // pool is not filled yet, so that won't fail
		});

		tx
	}

	pub(crate) fn tx(&self) -> &Sender<T> { &self.tx }

	pub(crate) fn rx(&self) -> &Receiver<T> { &self.rx }
}
