#!/bin/sh

watchexec -e py -r "python3 /runtime/launch_servers.py --verbose --port $PORT"
