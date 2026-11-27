#!/usr/bin/env bash
set -euo pipefail
exec python -m malow.training.evaluate "$@"
