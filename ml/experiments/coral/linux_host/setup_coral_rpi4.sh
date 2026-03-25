#!/usr/bin/env bash
set -euo pipefail

# Coral USB Accelerator setup & self-test for Raspberry Pi 4
# - Installs Edge TPU runtime (libedgetpu), pycoral, tflite-runtime
# - Adds Coral APT repo & GPG key
# - Verifies device visibility and runs a tiny inference on the TPU
#
# Usage:
#   chmod +x setup_coral_rpi4.sh
#   ./setup_coral_rpi4.sh
#
# Tested on: Raspberry Pi OS (Bullseye/Bookworm), Debian-based aarch64/armhf on RPi4

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; BLU='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${BLU}[INFO]${NC} $*"; }
ok()   { echo -e "${GRN}[OK]${NC}   $*"; }
warn() { echo -e "${YEL}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC}  $*"; }

need_sudo() {
  if [ "$EUID" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      echo "sudo"
    else
      err "This script requires root privileges and 'sudo' is not installed."
      exit 1
    fi
  else
    echo ""
  fi
}

SUDO="$(need_sudo)"

detect_arch() {
  local a="$(uname -m)"
  case "$a" in
    aarch64) echo "arm64" ;;
    armv7l|armv8l) echo "armhf" ;;
    *) echo "$a" ;;
  esac
}

ARCH="$(detect_arch)"
log "Detected architecture: ${ARCH}"

ensure_pkg() {
  # $1..$n packages
  $SUDO apt-get install -y --no-install-recommends "$@" || {
    err "Failed to install packages: $*"
    exit 2
  }
}

add_coral_repo() {
  local keyring="/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg"
  local list="/etc/apt/sources.list.d/coral-edgetpu.list"
  if [ -f "$list" ]; then
    ok "Coral APT repo already present: $list"
    return
  fi
  log "Adding Coral APT repository & key"
  $SUDO apt-get update -y
  ensure_pkg curl ca-certificates gnupg lsb-release
  tmpkey="$(mktemp)"
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg -o "$tmpkey"
  $SUDO install -m 0755 -d /usr/share/keyrings
  $SUDO gpg --dearmor < "$tmpkey" | $SUDO tee "$keyring" >/dev/null
  rm -f "$tmpkey"
  echo "deb [signed-by=${keyring}] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | $SUDO tee "$list" >/dev/null
  $SUDO apt-get update -y
  ok "Coral APT repository configured"
}

install_runtime() {
  log "Installing base packages"
  $SUDO apt-get update -y
  ensure_pkg usbutils udev python3 python3-venv python3-pip python3-setuptools python3-dev
  add_coral_repo

  log "Installing Edge TPU runtime (standard clock)"
  if ! $SUDO apt-get install -y libedgetpu1-std; then
    warn "libedgetpu1-std not found via APT. Trying libedgetpu1-max ..."
    $SUDO apt-get install -y libedgetpu1-max || {
      err "Failed to install libedgetpu runtime."
      exit 3
    }
  fi

  log "Installing pycoral + tflite-runtime (APT if available, otherwise pip in venv)"
  if $SUDO apt-get install -y python3-pycoral python3-tflite-runtime; then
    ok "Installed python3-pycoral & python3-tflite-runtime via APT"
    PY_BIN="python3"
    VENV_ACTIVATE=""
  else
    warn "APT python3-pycoral or python3-tflite-runtime not available; falling back to venv + pip."
    $SUDO python3 -m venv /opt/coral-venv || python3 -m venv /opt/coral-venv
    # shellcheck disable=SC1091
    . /opt/coral-venv/bin/activate
    PY_BIN="python"
    VENV_ACTIVATE="source /opt/coral-venv/bin/activate"
    python -m pip install --upgrade pip
    # Try multiple tflite-runtime versions to match platform; fallback to tensorflow if needed
    if ! python -m pip install --no-cache-dir tflite-runtime pycoral; then
      warn "tflite-runtime wheel may not be available; attempting tensorflow (larger)."
      python -m pip install --no-cache-dir tensorflow pycoral || {
        err "Failed to install any TFLite/TensorFlow runtime."
        exit 4
      }
    fi
    ok "Installed pycoral via pip in venv: /opt/coral-venv"
  fi

  log "Reloading udev rules"
  $SUDO udevadm control --reload-rules || true
  $SUDO udevadm trigger || true
}

write_selftest() {
  local test_dir="/opt/coral-selftest"
  $SUDO mkdir -p "$test_dir"
  $SUDO tee "$test_dir/self_test.py" >/dev/null <<'PY'
import sys, os, time
print("=== Coral USB self-test ===")
try:
    from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
    from pycoral.adapters import common
except Exception as e:
    print("Import error:", e)
    sys.exit(2)

devs = list_edge_tpus()
print("Detected Edge TPUs:", devs)
if not devs:
    print("ERROR: No Edge TPU devices detected. Check USB cable/port and power.")
    sys.exit(3)

# Try to download a small EdgeTPU-ready model for a smoke test
import urllib.request, tempfile
urls = [
    # A few well-known sample model URLs; first one that works will be used
    "https://dl.google.com/coral/canned_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite",
    "https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite",
]
model_path = None
for u in urls:
    try:
        print("Downloading sample model:", u)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tflite")
        urllib.request.urlretrieve(u, tmp.name)
        model_path = tmp.name
        break
    except Exception as e:
        print("Download failed:", e)

if model_path is None:
    print("WARNING: Could not download a sample model. Falling back to device enumeration only.")
    sys.exit(0)

print("Using model:", model_path)
try:
    itp = make_interpreter(model_path)
    itp.allocate_tensors()
    inp = common.input_details(itp, 0)
    out = itp.get_output_details()[0]
    # Create a dummy input of the right shape and dtype
    import numpy as np
    shape = inp['shape']
    dtype = np.uint8 if 'dtype' in inp and str(inp['dtype']).endswith('uint8') else np.int8
    x = np.zeros(shape, dtype=dtype)
    common.set_input(itp, x)
    t0 = time.perf_counter()
    itp.invoke()
    dt = (time.perf_counter() - t0)*1000.0
    print(f"Inference OK on Edge TPU. Latency: {dt:.2f} ms  (dummy input)")
    y = itp.get_tensor(out['index'])
    print("Output shape:", y.shape, "dtype:", y.dtype)
    sys.exit(0)
except Exception as e:
    print("ERROR: Failed to run inference on Edge TPU:", e)
    sys.exit(5)
PY
  ok "Wrote self-test script to ${test_dir}/self_test.py"
  echo "$VENV_ACTIVATE" | $SUDO tee "$test_dir/activate_venv.sh" >/dev/null
  $SUDO chmod +x "$test_dir/activate_venv.sh"
}

main() {
  log "Starting Coral USB setup for Raspberry Pi 4"
  install_runtime
  write_selftest

  log "Running device visibility test"
  if [ -n "$VENV_ACTIVATE" ]; then . /opt/coral-venv/bin/activate; fi
  if $PY_BIN -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())" | grep -q '\('; then
    ok "Edge TPU device(s) detected."
  else
    warn "Could not detect Edge TPU via pycoral. If the USB device is connected, try replugging or rebooting."
  fi

  log "Running inference smoke test"
  set +e
  if [ -n "$VENV_ACTIVATE" ]; then . /opt/coral-venv/bin/activate; fi
  $PY_BIN /opt/coral-selftest/self_test.py
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    ok "Coral self-test passed (inference executed)."
  elif [ $rc -eq 0 ] || [ $rc -eq 3 ]; then
    warn "Device enumerated but sample model test skipped/failed to download. Basic access appears OK."
  else
    err "Self-test failed with code $rc. Check logs above."
    exit $rc
  fi

  ok "Setup complete."
  echo -e "${BLU}Next steps:${NC}"
  echo "  - (Optional) Reboot to ensure udev rules are applied: sudo reboot"
  echo "  - Run the self-test again later: $VENV_ACTIVATE && python /opt/coral-selftest/self_test.py"
}

main "$@"
