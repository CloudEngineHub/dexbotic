#!/usr/bin/env bash
set -euo pipefail

plugin_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python -m pip install "${plugin_dir}"
