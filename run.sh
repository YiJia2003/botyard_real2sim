#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_python="${BOTYARD_REAL2SIM_PYTHON:-$project_dir/../IsaacLab/.venv/bin/python}"
if [[ ! -x "$runtime_python" ]]; then
  echo "找不到 Python：$runtime_python；请设置 BOTYARD_REAL2SIM_PYTHON。" >&2
  exit 1
fi
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd -- "$project_dir"

operation="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$operation" in
  replay|curves|sweep)
    exec "$runtime_python" -m "botyard_real2sim.cli.$operation" "$@"
    ;;
  test)
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    exec "$runtime_python" -m pytest -q "$@"
    ;;
  check)
    "$runtime_python" -m ruff check src scripts tests
    exec "$runtime_python" -m ruff format --check src scripts tests
    ;;
  help|-h|--help)
    echo '用法：./run.sh {replay|curves|sweep|test|check} [参数]'
    echo '示例：./run.sh replay --gui --joint FFJ2 --target-deg 20 --duration 2 --out logs/demo/sim.csv'
    ;;
  *)
    echo "未知操作：$operation；运行 ./run.sh --help 查看用法。" >&2
    exit 2
    ;;
esac
