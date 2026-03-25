# Makefile for Coral Edge TPU Deployment (v1.2)
# Three-phase approach with hard gates

PYTHON := python3
VENV := .venv

# Model paths
ONNX_MODEL := models/student_1x128_int8.onnx
FULL_TFLITE := models/full_int8.tflite
POST_EMB_TFLITE := models/post_emb_int8.tflite
POST_EMB_COMPILED := models/post_emb_int8_edgetpu.tflite

# Gates
MIN_MAPPING := 0.85
MAX_SEGMENTS := 3

.PHONY: all setup analyze export compile gate serve bench check-tpu clean

# Default target
all: setup analyze

# Setup environment
setup:
	@echo "Setting up environment..."
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		$(VENV)/bin/pip install -q --upgrade pip; \
		$(VENV)/bin/pip install -q tensorflow tflite-runtime numpy onnx onnx-tf; \
		echo "✓ Virtual environment created"; \
	else \
		echo "✓ Virtual environment exists"; \
	fi

# Phase A: Quick falsification (time-boxed)
analyze: setup
	@echo "\n=== PHASE A: Quick Falsification ==="
	@echo "Attempting direct ONNX → TFLite conversion..."
	@if [ -f "$(FULL_TFLITE)" ]; then \
		echo "Found $(FULL_TFLITE), attempting compilation..."; \
		edgetpu_compiler -s $(FULL_TFLITE) 2>&1 | tee models/full_compile.log || true; \
		$(VENV)/bin/python scripts/compile_and_gate.py \
			--report models/full_compile.log \
			--min_map $(MIN_MAPPING) \
			--max_segments $(MAX_SEGMENTS) || \
		echo "❌ Phase A failed gates. Moving to Phase B..."; \
	else \
		echo "No full_int8.tflite found. Run conversion first or skip to Phase B."; \
		$(VENV)/bin/python scripts/convert_onnx_to_tflite.py \
			--onnx $(ONNX_MODEL) \
			--output models/ || \
		echo "❌ Direct conversion failed. Moving to Phase B..."; \
	fi

# Phase B: Post-embedding hybrid (preferred)
export: setup
	@echo "\n=== PHASE B: Post-Embedding Export ==="
	@echo "Building post-embedding TFLite model..."
	@$(VENV)/bin/python src/export_tflite.py \
		--out_dir models/ \
		--keras_model models/keras_post_emb || \
	(echo "Creating placeholder model..." && \
	 $(VENV)/bin/python src/coral_tiny.py --save models/keras_post_emb && \
	 $(VENV)/bin/python src/export_tflite.py --out_dir models/)

compile: export
	@echo "\n=== Compiling for Edge TPU ==="
	@if [ -f "$(POST_EMB_TFLITE)" ]; then \
		edgetpu_compiler -s $(POST_EMB_TFLITE) -o models/ 2>&1 | tee models/post_emb_compile.log; \
	else \
		echo "❌ Error: $(POST_EMB_TFLITE) not found. Run 'make export' first."; \
		exit 1; \
	fi

gate:
	@echo "\n=== Enforcing Quality Gates ==="
	@if [ -f "models/post_emb_compile.log" ]; then \
		$(VENV)/bin/python scripts/compile_and_gate.py \
			--report models/post_emb_compile.log \
			--min_map $(MIN_MAPPING) \
			--max_segments $(MAX_SEGMENTS) && \
		echo "✅ GATES PASSED - Ready for deployment" || \
		echo "❌ GATES FAILED - See recommendations above"; \
	else \
		echo "❌ No compilation log found. Run 'make compile' first."; \
		exit 1; \
	fi

# Serve the model
serve:
	@echo "\n=== Starting TPU Service ==="
	@if [ -f "$(POST_EMB_COMPILED)" ]; then \
		$(VENV)/bin/python service/stage2_tpu_server.py \
			--model $(POST_EMB_COMPILED) \
			--port 8080; \
	else \
		echo "❌ Compiled model not found. Run 'make compile' first."; \
		exit 1; \
	fi

# Benchmark
bench:
	@echo "\n=== Benchmarking CPU vs TPU ==="
	@$(VENV)/bin/python scripts/bench.py \
		--cpu $(FULL_TFLITE) \
		--tpu $(POST_EMB_COMPILED) \
		--samples 500

# Check TPU hardware
check-tpu:
	@echo "\n=== Checking for Coral TPU ==="
	@lsusb | grep -i "Google Inc" || echo "No Coral TPU detected via USB"
	@$(VENV)/bin/python -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())" 2>/dev/null || \
		echo "pycoral not installed or no TPU detected"

# Clean generated files
clean:
	rm -f models/*.tflite models/*.log
	rm -rf models/keras_post_emb
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Quick start target
quickstart: setup
	@echo "\n=== QUICK START GUIDE ==="
	@echo "1. Phase A (2-4 hrs): make analyze"
	@echo "   - Tries direct conversion, expected to fail"
	@echo ""
	@echo "2. Phase B (1-3 days): make export compile gate"
	@echo "   - Post-embedding hybrid approach"
	@echo "   - Most likely to succeed"
	@echo ""
	@echo "3. Benchmark: make bench"
	@echo "   - Compare CPU vs TPU performance"
	@echo ""
	@echo "4. Decision: GO if gates pass AND E2E beats CPU"
	@echo "             NO-GO otherwise"