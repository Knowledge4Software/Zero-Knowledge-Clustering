#!/usr/bin/env bash
# Copy code (and optionally the dataset cache) to the research server. Run inside WSL.
# usage: bash campaigns/server/sync_to_server.sh [--data]   (from code/)
set -euo pipefail
HOST=${COMMITJL_HOST:-cloudfpsmobin}
items=(commitjl data experiments analysis tests campaigns pyproject.toml requirements.txt zk/bin/commitjl-query-linux)
[[ "${1:-}" == --data ]] && items+=(.cache/commitjl_data)
tar czf - --exclude=__pycache__ "${items[@]}" | ssh -o BatchMode=yes "$HOST" 'mkdir -p ~/commitjl && cd ~/commitjl && tar xzf - && du -sh . && ls'
