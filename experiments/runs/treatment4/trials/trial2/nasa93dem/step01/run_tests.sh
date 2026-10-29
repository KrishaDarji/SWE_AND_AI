#!/bin/sh
set -eu
PYTHONPATH=. python3 -m unittest discover -s tests -v
