.PHONY: build-dataset train-local train-cloud export-onnx service-build service-serve service-deploy test lint

build-dataset:
	python3 ml/data/build.py --recipe ml/data/recipes/pi_mix_v3.yaml --output ml/data/versions/pi_mix_v3

train-local:
	python3 ml/training/train_cls.py --config ml/training/config/model.yaml

train-cloud:
	python3 ml/cloud/submit.py --config ml/cloud/config_injection_only.yaml

export-onnx:
	python3 ml/export/export_onnx.py --model-path ml/models/bert-tiny-injection-only-20260317 --output-dir service/assets/

service-build:
	cd service && fastly compute build

service-serve:
	cd service && fastly compute serve

service-deploy:
	cd service && fastly compute deploy

test:
	uv run pytest ml/data/tests/ ml/training/tests/ ml/cloud/tests/ -v || true

lint:
	uv run ruff check .
	cd service && cargo clippy --all-targets -- -D warnings
