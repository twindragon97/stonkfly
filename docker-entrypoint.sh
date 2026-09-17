#!/bin/sh
set -e
cd /app
exec python -m stonkfly.service
