#!/bin/bash
# Installs a compiler that can build Zaphod (#embed needs Clang 19+ or GCC 15+)
# in Claude Code cloud sessions. Ubuntu 24.04 ships Clang 18 and GCC 13, so
# Clang 20 is installed and made the default clang/clang++, which is what the
# linux-clang preset uses.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if ! command -v clang++-20 >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q clang-20 cmake ninja-build
fi

update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-20   100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100
