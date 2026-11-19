#!/usr/bin/env bash
# Sets the project up on first use (or when requirements.txt changes), then starts the local page.
set -euo pipefail
cd "$(dirname "$0")"

want=$(sha256sum requirements.txt | cut -d' ' -f1)
have=$(cat .venv/.requirements-hash 2>/dev/null || true)
if [ ! -x .venv/bin/python ] || [ "$want" != "$have" ]; then
  echo "Installing packages into .venv (about a minute the first time)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q --only-binary :all: -r requirements.txt
  echo "$want" > .venv/.requirements-hash
fi

[ -f config.env ] || cp config.example.env config.env
exec .venv/bin/python server.py "$@"
