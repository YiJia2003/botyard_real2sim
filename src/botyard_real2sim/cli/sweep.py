#!/usr/bin/env python3
"""Run a reviewable one-parameter real2sim sweep; this is not automatic calibration.

Example (use the Python interpreter that can launch Isaac):
  python scripts/real2sim_sweep.py \
    --parameter tendon.cable_stiffness_n_per_m.FFJ2 --values 4000 8000 12000 \
    --out-dir /tmp/tendon-sweep --dry-run -- \
    --joint FFJ2 --target-deg 30 --duration 0.5 --config configs/real2sim/default.yaml

Each child runs in the same current directory and with this interpreter.
Failures are retained in manifest.json and logs; other runs continue. No
aggregate score, optimizer, or fitted-parameter claim is generated.

【中文说明】
职责：
  对一个参数做「组合扫描」：把同一个参数依次取 values 里的每个值，各生成一组回放命令
  （可选再加一组对比命令），逐组执行并把状态写进 manifest.json，供人事后判读与后续拟合。
  本脚本只负责「按值枚举 + 调用 + 记录」，本身不引入任何物理假设，也不做自动标定。

输入：
  --parameter  参数路径，语法形如 controller.NAME 或 tendon.NAME，可再跟 .JOINT 或 .INDEX
               （例如 tendon.cable_stiffness_n_per_m.FFJ2）；
  --values     同一参数的若干取值，必须是非空有限数（单位随该参数本身，如 N/m）；
  --out-dir    每组输出的根目录；
  --real       可选实机 CSV，给出后每组还会跑一次 real2sim_curves.py 生成对比图与指标；
  --dry-run    只生成计划，不真正启动子进程；
  --overwrite  允许复用已存在的输出目录（只清理计划内的已知产物）；
  --           之后的内容原样传给 real2sim_replay.py。

输出：
  <out-dir>/manifest.json（计划 + 每组状态 + 返回码 + 日志路径），以及每组子目录 run_NNN/
  下的 sim.csv、replay.log、（可选）comparison.html/json、comparison.log。

与 replay 的关系：
  本脚本是 real2sim_replay.py 的批量调用壳：参数是否真实存在、模型约束是否成立，都由子进程
  用它自己解析后的配置去检查；扫描脚本不复制也不改写回放逻辑。若给了 --real，则再调用
  real2sim_curves.py 生成实机—仿真对比。

退出码：
  0 表示所有组都没失败；1 表示至少有一组状态以 _failed 结尾（失败详情保留在 manifest/日志里）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# 参数路径白名单正则：必须以 controller. 或 tendon. 开头，随后是合法标识符；
# 之后允许任意多段 ".标识符 或 .数字"，用来指定具体关节（如 .FFJ2）或下标（如 .0）。
# 这样既能拦住拼写错误/路径注入，又不把参数名清单硬编码进本脚本。
_PARAMETER = re.compile(r"^(controller|tendon)\.[A-Za-z_]\w*(?:\.(?:[A-Za-z_]\w*|[0-9]+))*$")


def _option_value(arguments: list[str], option: str) -> str | None:
    """在参数列表里取出某个选项的取值，兼容 ``--opt value`` 与 ``--opt=value`` 两种写法。

    参数：
      arguments —— 待搜索的参数列表（这里是准备透传给回放脚本的参数）；
      option    —— 选项名，例如 "--config"。
    返回：该选项最后一次出现的取值；未出现则返回 None。
    异常：选项出现在列表末尾却没有跟随值、或下一个 token 以 ``--`` 开头（说明漏了值）
          时抛 ValueError。
    说明：只解析、不改写；同一选项重复出现时以最后一次为准。
    """
    found = None
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a value")
            found = arguments[index + 1]
        elif argument.startswith(option + "="):
            found = argument.split("=", 1)[1]
    return found


def plan_sweep(
    parameter: str,
    values: list[float],
    out_dir: str | Path,
    replay_args: list[str],
    *,
    real: str | Path | None = None,
    python_executable: str | None = None,
    comparison_align: str = "start",
) -> dict:
    """Build commands without importing Isaac or running any child process.

    Parameter path syntax is checked here; actual parameter existence and model
    constraints are checked by the replay child using its resolved config.

    【中文】构造扫描计划：为每个取值生成一条回放命令（以及可选的对比命令），不真的运行。

    参数：
      parameter   —— 参数路径，须匹配 _PARAMETER 的语法（controller.NAME / tendon.NAME 等）；
      values      —— 该参数的取值列表，单位随参数本身（如刚度 N/m、阻尼 N·s/m）；
      out_dir     —— 输出根目录，每组落在 run_001、run_002 …… 子目录；
      replay_args —— 原样透传给 real2sim_replay.py 的参数（--out 由本脚本接管）；
      real        —— 可选实机 CSV 路径；给出后每组追加一条 real2sim_curves.py 对比命令；
      python_executable —— 子进程解释器，默认取当前解释器（需能启动 Isaac 的那个）；
      comparison_align  —— 对比对齐模式，"start"（默认）或 "command_edge"。
    返回：计划字典（schema_version / parameter / values / runs / ...），可直接 JSON 序列化；
          runs 里每一组都带 index、value、路径、命令、初始 status="planned" 与日志路径。
    异常：参数语法不合法、values 为空或含非有限值、对比对齐模式非法、replay_args 里自带 --out、
          或混入 -h/--help/-- 时抛 ValueError。本函数只校验语法，不校验参数是否真实存在。
    """
    if not _PARAMETER.fullmatch(parameter):
        raise ValueError("parameter must be controller.NAME or tendon.NAME, optionally followed by .JOINT or .INDEX")
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("values must be nonempty finite numbers")
    if comparison_align not in {"start", "command_edge"}:
        raise ValueError("comparison_align must be start or command_edge")
    # ── 前置校验：先挡掉会与本脚本冲突的透传参数 ──
    replay_args = list(replay_args)
    if any(arg == "--out" or arg.startswith("--out=") for arg in replay_args):
        raise ValueError("replay --out is managed by the sweep; remove it from replay arguments")
    if any(arg in {"-h", "--help", "--"} for arg in replay_args):
        raise ValueError("replay arguments must contain runnable options, not help or another -- separator")
    # ── 路径解析：输出目录与脚本目录都取绝对路径，保证子进程在任意 cwd 下都能定位脚本 ──
    out_dir = Path(out_dir).resolve()
    # 解释器默认沿用当前解释器（即能够启动 Isaac 的那一个），可显式覆盖。
    executable = python_executable or sys.executable
    # 透传参数里的 --config 会被对比命令复用，从而让两组指标口径一致。
    config = _option_value(replay_args, "--config")
    # ── 组合生成：每个取值对应一个 run_NNN 子目录和一组命令，顺序与 values 一致 ──
    runs = []
    for index, value in enumerate(values, 1):
        # 目录编号从 1 开始、补零到三位：run_001、run_002 ……，与 manifest 里的 index 对齐。
        run_dir = out_dir / f"run_{index:03d}"
        csv_path = run_dir / "sim.csv"
        # 回放命令 = 解释器 + real2sim_replay.py + 透传参数 + --set 参数=值 + --out。
        # 取值用 17 位有效数字格式化，避免扫描过程中因打印精度丢掉差异。
        command = [
            executable,
            "-m",
            "botyard_real2sim.cli.replay",
            *replay_args,
            "--set",
            f"{parameter}={value:.17g}",
            "--out",
            str(csv_path),
        ]
        compare_command = None
        # 可选对比命令：给出 --real 时，每组回放结束再用 real2sim_curves.py 与实机对照一次；
        # 它只在本组回放成功之后才会被执行（见 execute_plan 的阶段顺序）。
        if real is not None:
            compare_command = [
                executable,
                "-m",
                "botyard_real2sim.cli.curves",
                "--real",
                str(Path(real).resolve()),
                "--sim",
                str(csv_path),
                "--align",
                comparison_align,
                "--out",
                str(run_dir / "comparison.html"),
                "--report",
                str(run_dir / "comparison.json"),
            ]
            # 透传了 --config 就让对比脚本用同一份配置，保证单位换算与通道选择一致。
            if config is not None:
                compare_command += ["--config", config]
        runs.append(
            {
                "index": index,
                "value": value,
                "run_dir": str(run_dir),
                "sim_csv": str(csv_path),
                "replay_command": command,
                "comparison_command": compare_command,
                "status": "planned",
                "replay_log": str(run_dir / "replay.log"),
                "comparison_log": str(run_dir / "comparison.log"),
            }
        )
    # ── 计划内容：只描述「做了什么、在哪、状态如何」，不给出综合评分或拟合结论 ──
    return {
        "schema_version": 1,
        "purpose": "parameter sweep, not automatic calibration or fitted parameters",
        "parameter": parameter,
        "values": list(values),
        "out_dir": str(out_dir),
        "working_directory": str(Path.cwd()),
        "python_executable": executable,
        "real_csv": str(Path(real).resolve()) if real is not None else None,
        "comparison_alignment": comparison_align,
        "runs": runs,
    }


def _save(plan: dict) -> None:
    """把计划写回 <out-dir>/manifest.json，采用「先写临时文件再原子替换」的方式。

    参数：plan —— 计划字典（含 out_dir 与 runs 状态）。
    返回：无。
    为什么这样写：扫描过程中每完成一个阶段都会落盘一次，若中途被打断，原子替换能保证
    manifest.json 要么是旧的完整版本、要么是新的完整版本，不会留下半个 JSON。
    异常：磁盘不可写等 OSError 直接向上抛出（由 main 统一转成参数错误）。
    """
    destination = Path(plan["out_dir"]) / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def execute_plan(
    plan: dict,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    runner: Callable = subprocess.run,
) -> dict:
    """Execute argument lists with captured logs; record failures and continue.

    An existing nonempty output directory is refused unless overwrite=True.
    Overwrite clears only known artifacts of each planned run, preventing a
    failed child from accidentally reusing an old simulator CSV/report.

    【中文】逐组执行计划中的命令，把输出写进每个 stage 对应的日志；失败就记录并继续下一组。

    参数：
      plan      —— plan_sweep 产出的计划字典（会被就地更新状态并写入 manifest.json）；
      dry_run   —— True 时只把状态标成 "planned"，不创建子目录、不启动任何子进程；
      overwrite —— True 时允许复用已存在的输出目录，并且只删除计划内的已知产物
                   （sim.csv / comparison.html / comparison.json / replay.log / comparison.log）；
      runner    —— 真正执行命令的函数，默认 subprocess.run，便于测试替换。
    返回：更新后的 plan（每组含 status、各 stage 的返回码与错误说明）。
    异常：输出目录已存在且非空（且未授予 overwrite）时抛 ValueError，避免覆盖他人结果。
    失败处理：某个 stage 返回非零码、未产出预期文件、或启动时 OSError，则该组停止继续执行
              后续 stage，状态记为 "<stage>_failed"；其余各组照常运行。
    """
    out_dir = Path(plan["out_dir"])
    if (
        out_dir.exists()
        and (not out_dir.is_dir() or any(out_dir.iterdir()))
        and (not overwrite or not out_dir.is_dir())
    ):
        raise ValueError(f"output collision: {out_dir}; use --overwrite to reuse a directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    # ── 第一轮：重置本组状态，清掉上一次运行遗留的错误信息 ──
    plan["dry_run"] = dry_run
    for run in plan["runs"]:
        run["status"] = "planned" if dry_run else "pending"
        for key in ("error", "replay_returncode", "comparison_returncode"):
            run.pop(key, None)
    _save(plan)
    if dry_run:
        return plan
    # ── 第二轮：逐组执行（dry_run 时不会走到这里）──
    for run in plan["runs"]:
        run_dir = Path(run["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        # overwrite 只精确清理本脚本已知的产物文件：避免本轮失败的子进程误读到上一轮遗留的
        # sim.csv / comparison.json，也避免误删用户放在同一目录里的其它文件。
        if overwrite:
            for name in ("sim.csv", "comparison.html", "comparison.json", "replay.log", "comparison.log"):
                artifact = run_dir / name
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
        # ── 阶段列表：先回放（必须产出 sim.csv），再可选地做对比（必须产出 comparison.json）──
        stages = [("replay", run["replay_command"], Path(run["sim_csv"]))]
        if run["comparison_command"] is not None:
            stages.append(("comparison", run["comparison_command"], run_dir / "comparison.json"))
        # 逐个阶段执行：先把状态记成 "<stage>_running" 并落盘，便于运行中观察进度。
        for stage, command, expected in stages:
            run["status"] = f"{stage}_running"
            _save(plan)
            try:
                with Path(run[f"{stage}_log"]).open("w", encoding="utf-8") as log:
                    result = runner(
                        command, stdout=log, stderr=subprocess.STDOUT, cwd=plan["working_directory"], check=False
                    )
                # 子进程的 stdout/stderr 合并写入该阶段日志，避免刷屏且保留排查依据。
                run[f"{stage}_returncode"] = result.returncode
                # 非零返回码即失败：记录原因并跳出本组，不再执行后续阶段。
                if result.returncode != 0:
                    run["status"] = f"{stage}_failed"
                    run["error"] = f"{stage} exited with code {result.returncode}; see {stage}_log"
                    break
                # 产物校验：返回 0 但没产出预期文件同样按失败处理，避免把空结果当成成功。
                if not expected.is_file() or (stage == "comparison" and not (run_dir / "comparison.html").is_file()):
                    run["status"] = f"{stage}_failed"
                    run["error"] = f"{stage} returned success but did not produce {expected}"
                    break
            # 子进程根本没能启动（解释器不存在、权限不足等）：记下原因并追加写入日志。
            except OSError as exc:
                run["status"] = f"{stage}_failed"
                run["error"] = str(exc)
                with Path(run[f"{stage}_log"]).open("a", encoding="utf-8") as log:
                    log.write(f"\nFailed to launch: {exc}\n")
                break
            finally:
                _save(plan)
        # for … else：只有所有阶段都没有 break（即全部成功）时才会执行到这里。
        else:
            run["status"] = "completed"
        _save(plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析参数 → 生成扫描计划 → 逐组执行 → 打印最终计划并给出退出码。

    参数：argv —— 参数列表；None 表示取 sys.argv[1:]，便于测试直接传入。
    返回：0 表示所有组都没有失败；1 表示至少有一组状态以 "_failed" 结尾。
    异常/退出：参数取值非法（如 --real 指向的文件不存在）由 parser.error 报错并以退出码 2 结束。
    要点：
      - "--" 之后的内容原样透传给 real2sim_replay.py，本脚本只额外接管 --set 与 --out；
      - 参数是否真实存在、模型约束是否成立，由回放子进程用自己解析后的配置去判断；
      - 无论成功失败，最终计划都会以 JSON 打印到 stdout，并已写入 manifest.json。
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--real", type=Path)
    parser.add_argument("--comparison-align", choices=("start", "command_edge"), default="start")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("replay_args", nargs=argparse.REMAINDER, help="replay options after --")
    args = parser.parse_args(argv)
    # argparse.REMAINDER 会把 "--" 及其后内容一起留下；这里先去掉开头的分隔符再透传。
    replay_args = args.replay_args
    if replay_args and replay_args[0] == "--":
        replay_args = replay_args[1:]
    try:
        # 实机 CSV 若给出就必须存在；否则每组对比都会静默失败，不如提前报错。
        if args.real is not None and not args.real.is_file():
            raise ValueError(f"real capture does not exist: {args.real}")
        # ── 计划阶段：只做语法/路径校验并生成命令，不启动任何子进程 ──
        plan = plan_sweep(
            args.parameter,
            args.values,
            args.out_dir,
            replay_args,
            real=args.real,
            comparison_align=args.comparison_align,
        )
        # ── 执行阶段：--dry-run 时只写 manifest；真正的失败不会抛异常，而是记进每组状态 ──
        execute_plan(plan, dry_run=args.dry_run, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # 最终计划打印到 stdout，便于人工或下游脚本直接读取（与 manifest.json 内容一致）。
    print(json.dumps(plan, indent=2, allow_nan=False))
    # 退出码：任一组以 _failed 结尾即返回 1，全部成功返回 0。
    return int(any(run["status"].endswith("_failed") for run in plan["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
