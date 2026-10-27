#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./start_mac_linux.sh once first."; exit 1; }
.venv/bin/python -m pytest
