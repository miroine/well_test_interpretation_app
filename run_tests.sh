#!/usr/bin/env bash
cd "$(dirname "$0")"
exec python run_tests.py "$@"
