python-lint:
	isort *.py
	black *.py
	flake8 *.py


rs-fmt:
	rustfmt +nightly src/*.rs

rs-check:
	cargo check --release --all-features --all-targets

rs-clippy:
	cargo clippy --release --all-features --all-targets -- -D warnings

rust-lint: rs-fmt rs-check rs-clippy
