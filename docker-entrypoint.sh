#!/bin/sh
set -e

cd /app

echo "======================================"
echo " Stonkfly Docker"
echo " Paper trading + Web Dashboard"
echo "======================================"

mkdir -p /app/data
mkdir -p /app/runs/paper
mkdir -p /app/dashboard/cache

echo "[stonkfly] Preparando/verificando dataset..."
python -m stonkfly prepare

echo "[stonkfly] Iniciando dashboard..."
HOST="${HOST:-0.0.0.0}" \
PORT="${PORT:-8765}" \
STONKFLY_DATA="${STONKFLY_DATA:-/app/data}" \
STONKFLY_RUN="${STONKFLY_RUN:-/app/runs/paper}" \
STONKFLY_VIZ_CACHE="${STONKFLY_VIZ_CACHE:-/app/dashboard/cache}" \
python dashboard/server.py "${STONKFLY_RUN:-/app/runs/paper}" &

DASHBOARD_PID=$!

echo "[stonkfly] Dashboard PID: ${DASHBOARD_PID}"
echo "[stonkfly] Dashboard: http://0.0.0.0:${PORT:-8765}"

cleanup() {
    echo "[stonkfly] Deteniendo servicios..."

    if kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        kill "$DASHBOARD_PID"
    fi
}

trap cleanup INT TERM EXIT

echo "[stonkfly] Iniciando paper trading..."

python -m stonkfly run --out /app/runs/paper &

STONKFLY_PID=$!

wait "$STONKFLY_PID"
STONKFLY_EXIT=$?

echo "[stonkfly] Stonkfly terminó con código ${STONKFLY_EXIT}"

exit "$STONKFLY_EXIT"
