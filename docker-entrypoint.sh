#!/bin/sh
set -e

cd /app

echo "======================================"
echo " Stonkfly Docker"
echo " Mode: PAPER TRADING"
echo "======================================"

mkdir -p /app/data
mkdir -p /app/runs

echo "[stonkfly] Preparando/verificando dataset..."
python -m stonkfly prepare

echo "[stonkfly] Iniciando paper trading..."
exec python -m stonkfly run
