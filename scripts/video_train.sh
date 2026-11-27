#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python}" -m malow.training.train "$@"
