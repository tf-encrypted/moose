#!/bin/sh

watchexec -e py -r "python3 /runtime/worker.py --verbose --port $PORT"
