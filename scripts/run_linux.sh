#!/usr/bin/env bash
set -euo pipefail

# Move to repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================================="
echo "  Chip Anomaly Detection System - Quick Launcher (Linux/macOS)"
echo "=============================================================="

PYTHON_BIN=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_BIN="$cand"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] Không tìm thấy Python 3! Vui lòng cài đặt Python 3.10 - 3.12."
    exit 1
fi

echo "Sử dụng trình thông dịch: $($PYTHON_BIN --version)"

if [[ ! -x ".venv/bin/python" ]]; then
    echo "[1/3] Đang tạo môi trường ảo (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

echo "[2/3] Kiểm tra và cài đặt thư viện cần thiết..."
if ! .venv/bin/python -c "import streamlit, torch, torchvision, cv2, scipy, yaml" >/dev/null 2>&1; then
    .venv/bin/python -m pip install --upgrade pip -q
    echo "Đang cài đặt các gói từ requirements.txt (lần đầu có thể mất vài phút)..."
    .venv/bin/pip install -r requirements.txt
fi

if [[ ! -f "weights/patchcore_best.ckpt" ]]; then
    echo "[WARNING] Không tìm thấy weights/patchcore_best.ckpt! Vui lòng kiểm tra lại bản clone."
fi

echo "[3/3] Khởi động giao diện Web UI tại http://localhost:8501 ..."
exec .venv/bin/streamlit run src/ui/app.py
