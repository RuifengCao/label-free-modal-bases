#!/bin/bash
# Wait until FILE contains PATTERN, then run the rest of the arguments as one shell command line.
f=$1; pat=$2; shift 2
while ! grep -q "$pat" "$f" 2>/dev/null; do sleep 15; done
eval "$@"
