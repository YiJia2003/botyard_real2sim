"""Operate the local <-> remote real2sim data bridge.

This CLI is deliberately simulator-free.  It builds canonical bundles, verifies
them locally, uploads them to a remote staging directory, asks the remote
verifier to mark them ready, notifies the remote DSH agent, and reads back the
remote result packet.

Typical local use (system Python is enough)::

    python3 scripts/real2sim_bridge.py doctor
    python3 scripts/real2sim_bridge.py build-selftest
    python3 scripts/real2sim_bridge.py publish r2s-...-selftest
    python3 scripts/real2sim_bridge.py notify r2s-...-selftest --timeout-s 3600
    python3 scripts/real2sim_bridge.py results show r2s-...-selftest

中文说明：
  本地只负责打包、校验、上传、通知和读取远程结果；仿真/拟合永远在远程。
  上传先进入 `.staging/<bundle>.partial-*`，远程复核哈希与 manifest 后才
  写 READY.json 并原子改名为正式目录。远程结果同样先 staging 再发布。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from botyard_real2sim import bridge
from botyard_real2sim.bridge import (
    BridgeConfig,
    BridgeError,
    DshClient,
    auto_bundle_id,
    ensure_bundle_id,
    export_capture,
    load_config,
    make_selftest_bundle,
    read_json,
    save_config,
    validate_dataset_bundle,
    verify_bundle_integrity,
)

DEFAULT_CAPTURE_JOINTS = bridge.SUPPORTED_CAPTURE_JOINTS
DEFAULT_BRIDGE_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "bridge"


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _mapping(values: Sequence[str], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise BridgeError(f"{name} entries must be NAME=VALUE")
        try:
            value = float(raw)
        except ValueError as error:
            raise BridgeError(f"{name}[{key}] must be a number") from error
        if not (value > 0):
            raise BridgeError(f"{name}[{key}] must be positive")
        result[key] = value
    return result


def _tolerances(args: argparse.Namespace) -> dict[str, float]:
    values = _mapping(args.tolerance or [], "tolerance")
    for key in ("angle_deg", "motor_angle_deg", "tension_N"):
        if key not in values:
            raise BridgeError(f"missing explicit tolerance {key}=VALUE")
    return values


def _resolve_bundle(path_or_id: str, config: BridgeConfig) -> Path:
    candidate = Path(path_or_id).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    root = config.local_experiment
    if not root.is_absolute():
        root = Path.cwd() / root
    return (root / ensure_bundle_id(path_or_id)).resolve()


def _state_dir(config: BridgeConfig) -> Path:
    root = config.local_experiment
    if not root.is_absolute():
        root = Path.cwd() / root
    return root / ".bridge" / "state"


def _read_state(config: BridgeConfig, bundle_id: str) -> dict[str, Any]:
    path = _state_dir(config) / f"{bundle_id}.json"
    return read_json(path) if path.is_file() else {}


def _write_state(config: BridgeConfig, bundle_id: str, value: Mapping[str, Any]) -> Path:
    path = _state_dir(config) / f"{bundle_id}.json"
    bridge.write_json_atomic(path, dict(value))
    return path


def _remote_bundle_path(config: BridgeConfig, bundle_id: str) -> str:
    return f"{config.remote_experiment}/{bundle_id}"


def _remote_result_path(config: BridgeConfig, bundle_id: str) -> str:
    return f"{config.remote_results}/{bundle_id}"


def _remote_staging_path(config: BridgeConfig, bundle_id: str, nonce: str) -> str:
    return f"{config.remote_experiment}/.staging/{bundle_id}.partial-{nonce}"


def _sync_bridge_scripts(config: BridgeConfig, script_dir: Path) -> None:
    if not script_dir.is_dir():
        raise BridgeError(f"bridge script directory not found: {script_dir}")
    bridge.sync_bridge_scripts(config, script_dir)


def _remote_verify_command(config: BridgeConfig, bundle_id: str, staging: str, *, replace: bool) -> str:
    args = [
        "python3",
        f"{config.remote_root}/real2sim/bridge/remote_verify.py",
        "--project-root",
        str(config.remote_root),
        "--bundle-id",
        bundle_id,
        "--staging",
        staging,
    ]
    if replace:
        args.append("--replace")
    return " ".join(bridge.shlex.quote(part) for part in args)


def _publish(
    config: BridgeConfig,
    bundle_dir: Path,
    *,
    replace: bool,
    skip_local_dataset: bool,
    sync_scripts: bool,
) -> dict[str, Any]:
    bundle_id = ensure_bundle_id(bundle_dir.name)
    integrity = verify_bundle_integrity(bundle_dir)
    if not integrity["ok"]:
        raise BridgeError("local bundle integrity failed: " + "; ".join(integrity["errors"]))
    dataset_report: dict[str, Any] | None = None
    if not skip_local_dataset:
        dataset_report = validate_dataset_bundle(bundle_dir)
        if not dataset_report.get("data_ready"):
            reasons = dataset_report.get("errors") or dataset_report.get("blocking_reasons") or []
            raise BridgeError(
                "local dataset is not strict-ready; fix the bundle or pass --skip-local-dataset for inspection: "
                + json.dumps(reasons, ensure_ascii=False)
            )
    bridge.ensure_remote_layout(config)
    if sync_scripts:
        _sync_bridge_scripts(config, DEFAULT_BRIDGE_SCRIPT_DIR)
    nonce = uuid.uuid4().hex[:8]
    staging = _remote_staging_path(config, bundle_id, nonce)
    remote_parent = f"{config.remote_experiment}/.staging"
    bridge.run_ssh(config, f"mkdir -p {bridge.shlex.quote(remote_parent)}")
    bridge.run_rsync(config, bundle_dir, staging, delete=True)
    verify = bridge.run_ssh(config, _remote_verify_command(config, bundle_id, staging, replace=replace), timeout=300)
    try:
        receipt = json.loads(verify.stdout[verify.stdout.find("{") :])
    except (ValueError, json.JSONDecodeError) as error:
        raise BridgeError(f"remote verifier returned non-JSON output:\n{verify.stdout}\n{verify.stderr}") from error
    if not receipt.get("ok"):
        raise BridgeError("remote verification failed: " + json.dumps(receipt, ensure_ascii=False))
    state = {
        "schema_version": bridge.SCHEMA_VERSION,
        "protocol": bridge.BRIDGE_PROTOCOL,
        "bundle_id": bundle_id,
        "local_bundle": str(bundle_dir),
        "remote_bundle": _remote_bundle_path(config, bundle_id),
        "remote_staging": staging,
        "published_at_utc": bridge.utc_now(),
        "remote_receipt": receipt,
        "local_integrity": integrity,
        "local_dataset_report": dataset_report,
    }
    _write_state(config, bundle_id, state)
    return state


def _remote_verify_existing(config: BridgeConfig, bundle_id: str) -> dict[str, Any]:
    result = bridge.run_ssh(
        config,
        f"test -f {bridge.shlex.quote(_remote_bundle_path(config, bundle_id) + '/READY.json')} && "
        f"cat {bridge.shlex.quote(_remote_bundle_path(config, bundle_id) + '/READY.json')}",
        check=False,
    )
    if result.returncode != 0:
        raise BridgeError(f"remote bundle {bundle_id} has no READY.json")
    return json.loads(result.stdout)


def _resolve_session(client: DshClient, config: BridgeConfig, explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    if config.dsh_session and config.dsh_session != "auto":
        return config.dsh_session
    return str(client.latest_session(str(config.remote_root))["sessionId"])


def _notify(
    config: BridgeConfig,
    bundle_id: str,
    *,
    session: str | None,
    wait: bool,
    timeout_s: float,
) -> dict[str, Any]:
    if not config.dsh_token:
        raise BridgeError(
            "no remote DSH token configured; set R2S_REMOTE_DSH_TOKEN in "
            f"{bridge.default_config_path()} or the environment"
        )
    local_bundle = config.local_experiment / bundle_id
    task = read_json(local_bundle / "TASK.json")
    client = DshClient(config.dsh_url, config.dsh_token)
    session_id = _resolve_session(client, config, session)
    before_seq = client.session_cursor(session_id)
    prompt = bridge.build_remote_task_prompt(
        bundle_id=bundle_id,
        remote_bundle_dir=_remote_bundle_path(config, bundle_id),
        remote_result_dir=_remote_result_path(config, bundle_id),
        intent=str(task.get("intent") or "calibration"),
        split=str(task.get("split") or "train"),
        tolerances=task.get("tolerances") or {},
        notes=str(task.get("notes") or ""),
    )
    receipt = client.prompt(session_id, prompt)
    final_text = _wait_for_turn(client, session_id, before_seq, timeout_s=timeout_s) if wait else ""
    state = _read_state(config, bundle_id)
    state.update(
        {
            "notified_at_utc": bridge.utc_now(),
            "dsh_session": session_id,
            "dsh_prompt_receipt": receipt,
            "dsh_final_text": final_text,
        }
    )
    _write_state(config, bundle_id, state)
    return {"session_id": session_id, "prompt_receipt": receipt, "final_text": final_text}


def _wait_for_turn(client: DshClient, session_id: str, before_seq: int, *, timeout_s: float) -> str:
    deadline = time.monotonic() + max(5.0, timeout_s)
    running_seen = False
    settle_deadline = time.monotonic() + min(10.0, timeout_s)
    while time.monotonic() < deadline:
        try:
            running = client.session_running(session_id)
        except BridgeError:
            running = False
        if running:
            running_seen = True
        if running_seen and not running:
            break
        if not running_seen and time.monotonic() > settle_deadline:
            break
        time.sleep(2.0)
    try:
        return client.latest_assistant_text(session_id, since_seq=before_seq)
    except BridgeError:
        return ""


def _say(
    config: BridgeConfig,
    text: str,
    *,
    session: str | None,
    wait: bool,
    timeout_s: float,
) -> dict[str, Any]:
    if not config.dsh_token:
        raise BridgeError(
            "no remote DSH token configured; set R2S_REMOTE_DSH_TOKEN in "
            f"{bridge.default_config_path()} or the environment"
        )
    client = DshClient(config.dsh_url, config.dsh_token)
    session_id = _resolve_session(client, config, session)
    before_seq = client.session_cursor(session_id)
    receipt = client.prompt(session_id, text)
    final_text = _wait_for_turn(client, session_id, before_seq, timeout_s=timeout_s) if wait else ""
    return {"session_id": session_id, "prompt_receipt": receipt, "final_text": final_text}


def _result_show(config: BridgeConfig, bundle_id: str, *, raw: bool) -> dict[str, Any]:
    remote = _remote_result_path(config, bundle_id)
    ready = bridge.run_ssh(config, f"cat {bridge.shlex.quote(remote + '/RESULT_READY.json')}", check=False)
    if ready.returncode != 0:
        raise BridgeError(f"remote result {bundle_id} is not published (no RESULT_READY.json)")
    result = json.loads(bridge.run_ssh(config, f"cat {bridge.shlex.quote(remote + '/RESULT.json')}").stdout)
    request_text = bridge.run_ssh(config, f"cat {bridge.shlex.quote(remote + '/next_data_request.json')}", check=False)
    request = json.loads(request_text.stdout) if request_text.returncode == 0 and request_text.stdout.strip() else None
    report = bridge.run_ssh(config, f"cat {bridge.shlex.quote(remote + '/report.md')}", check=False)
    payload = {
        "bundle_id": bundle_id,
        "remote_result_dir": remote,
        "ready": json.loads(ready.stdout),
        "result": result,
        "next_data_request": request,
        "report_excerpt": "\n".join(report.stdout.splitlines()[:120]) if report.returncode == 0 else "",
    }
    if raw:
        _print_json(payload)
        return payload
    print(f"bundle: {bundle_id}")
    print(f"status: {result.get('status')}  scope: {result.get('claims_scope')}")
    print(f"remote: {remote}")
    summary = result.get("summary") or result.get("diagnosis", {}).get("summary")
    if summary:
        print("\nsummary:\n" + str(summary))
    if request:
        print("\nnext data request:")
        print(f"  status: {request.get('status')}")
        print(f"  reason: {request.get('reason')}")
        for experiment in request.get("experiments") or request.get("requests") or []:
            print(
                "  - "
                + json.dumps(
                    {
                        key: experiment.get(key)
                        for key in (
                            "id",
                            "joints",
                            "profile",
                            "center_deg",
                            "amplitude_deg",
                            "speed_cap_deg_s",
                            "directions",
                            "dwell_s",
                            "repeats",
                            "purpose",
                        )
                        if experiment.get(key) is not None
                    },
                    ensure_ascii=False,
                )
            )
    if payload["report_excerpt"]:
        print("\nreport excerpt:\n" + payload["report_excerpt"])
    return payload


def _result_pull(config: BridgeConfig, bundle_id: str, *, replace: bool) -> Path:
    remote = _remote_result_path(config, bundle_id)
    target = config.local_experiment / bundle_id / "results"
    if target.exists() and not replace:
        raise BridgeError(f"local result directory already exists (use --replace): {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    bridge.run_rsync_download(config, remote, target, delete=False)
    return target


def _status(config: BridgeConfig, bundle_id: str | None, *, as_json: bool) -> dict[str, Any]:
    script = (
        "set -e; "
        f"cd {bridge.shlex.quote(str(config.remote_root))}; "
        "echo '== remote experiment'; ls -1 real2sim/experiment 2>/dev/null || true; "
        "echo '== experiment staging'; ls -1 real2sim/experiment/.staging 2>/dev/null || true; "
        "echo '== results'; ls -1 real2sim/results 2>/dev/null || true; "
        "echo '== results staging'; ls -1 real2sim/results/.staging 2>/dev/null || true"
    )
    listing = bridge.run_ssh(config, script, check=False)
    state = _read_state(config, bundle_id) if bundle_id else {}
    payload: dict[str, Any] = {
        "remote_host": config.remote_host,
        "remote_root": str(config.remote_root),
        "ssh_ok": listing.returncode == 0,
        "listing": listing.stdout,
        "local_state": state,
    }
    if bundle_id:
        payload["local_bundle"] = str(config.local_experiment / bundle_id)
        payload["remote_bundle"] = _remote_bundle_path(config, bundle_id)
        payload["remote_result"] = _remote_result_path(config, bundle_id)
        payload["remote_ready"] = bridge.run_ssh(
            config,
            f"test -f {bridge.shlex.quote(_remote_bundle_path(config, bundle_id) + '/READY.json')} "
            "&& echo yes || echo no",
            check=False,
        ).stdout.strip()
        payload["result_ready"] = bridge.run_ssh(
            config,
            f"test -f {bridge.shlex.quote(_remote_result_path(config, bundle_id) + '/RESULT_READY.json')} "
            "&& echo yes || echo no",
            check=False,
        ).stdout.strip()
    if config.dsh_token and bundle_id:
        try:
            client = DshClient(config.dsh_url, config.dsh_token)
            session_id = _resolve_session(client, config, None)
            payload["dsh_session"] = session_id
            payload["dsh_running"] = client.session_running(session_id)
        except BridgeError as error:
            payload["dsh_error"] = str(error)
    if as_json:
        _print_json(payload)
    else:
        print(f"remote: {config.remote_host}:{config.remote_root}")
        print(f"ssh: {'ok' if payload['ssh_ok'] else 'FAILED'}")
        print(payload["listing"].rstrip())
        if bundle_id:
            print(f"bundle ready: {payload['remote_ready']}")
            print(f"result ready: {payload['result_ready']}")
            if payload.get("dsh_session"):
                print(f"dsh session: {payload['dsh_session']} running={payload.get('dsh_running')}")
            if payload.get("dsh_error"):
                print(f"dsh error: {payload['dsh_error']}")
    return payload


def _doctor(config: BridgeConfig) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("SSH key", config.remote_key.is_file(), str(config.remote_key)))
    if config.remote_key.is_file():
        mode = config.remote_key.stat().st_mode & 0o777
        checks.append(("SSH key mode 600", mode == 0o600, oct(mode)))
    remote = bridge.run_ssh(
        config,
        "set -e; whoami; hostname; "
        f"test -d {bridge.shlex.quote(str(config.remote_root))}; "
        f"test -x {bridge.shlex.quote(str(config.remote_root / 'run.sh'))}; "
        f"mkdir -p {bridge.shlex.quote(config.remote_experiment)} {bridge.shlex.quote(config.remote_results)} "
        f"{bridge.shlex.quote(config.remote_bridge)}; echo OK",
        check=False,
        timeout=30,
    )
    checks.append(("SSH reachable", remote.returncode == 0, remote.stdout.strip() or remote.stderr.strip()))
    checks.append(("rsync", shutil.which("rsync") is not None, shutil.which("rsync") or "not found"))
    if config.dsh_token:
        try:
            client = DshClient(config.dsh_url, config.dsh_token)
            session = client.latest_session(str(config.remote_root))
            checks.append(("DSH token", True, "accepted"))
            checks.append(("DSH session", True, f"{session.get('sessionId')} running={session.get('running')}"))
        except BridgeError as error:
            checks.append(("DSH token", False, str(error)))
    else:
        checks.append(("DSH token", False, "not configured (notify/watch unavailable)"))
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")
    return 0 if all(ok for _, ok, _ in checks if _ != "DSH token") else 1


def _plan_requests(request_path: Path, out_dir: Path, *, r2s_root: Path, effort_raw: int) -> dict[str, Any]:
    request = read_json(request_path)
    if not isinstance(request, dict):
        raise BridgeError("next_data_request.json root must be an object")
    experiments = request.get("experiments") or request.get("requests") or []
    if not isinstance(experiments, list) or not experiments:
        raise BridgeError("next_data_request.json has no experiments/requests")
    preparation_message = (
        "request requires local plan preparation: the hardware-host agent must prepare and "
        "validate the executable plan, then clear local_preparation_required before plan-requests"
    )
    if request.get("local_preparation_required") is True:
        raise BridgeError(preparation_message)
    seen_slugs: set[str] = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise BridgeError(f"experiments[{index}] must be an object")
        if experiment.get("local_preparation_required") is True:
            raise BridgeError(f"experiments[{index}]: {preparation_message}")
        slug = bridge.slug(str(experiment.get("id") or f"request-{index}"))
        if slug in seen_slugs:
            raise BridgeError("experiment IDs collide after filename normalization; use distinct IDs")
        seen_slugs.add(slug)
    out_dir = out_dir.expanduser().resolve()
    r2s_root = r2s_root.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"cd {bridge.shlex.quote(str(r2s_root))}", ""]
    plan: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise BridgeError(f"experiments[{index}] must be an object")
        joints = experiment.get("joints") or ([experiment["joint"]] if experiment.get("joint") else [])
        if not joints:
            raise BridgeError(f"experiments[{index}] needs joints")
        execution_mode = experiment.get("execution_mode", "paired" if "pair_plan" in experiment else "separate")
        if execution_mode not in ("separate", "paired"):
            raise BridgeError(f"experiments[{index}] has unsupported execution_mode")
        if execution_mode == "separate" and "pair_plan" in experiment:
            raise BridgeError("pair_plan requires paired execution; it cannot be split silently")
        if execution_mode == "paired":
            pair_plan = experiment.get("pair_plan")
            if (
                list(joints) != ["MFJ2", "MFJ3"]
                or not isinstance(pair_plan, dict)
                or pair_plan.get("schema") != "real2sim-pair-plan-v1"
                or pair_plan.get("joint_order") != ["MFJ2", "MFJ3"]
                or not isinstance(pair_plan.get("points"), list)
                or len(pair_plan["points"]) < 2
                or not isinstance(pair_plan.get("metadata", {}), dict)
            ):
                raise BridgeError("paired requests require both middle joints and a real2sim-pair-plan-v1 payload")
            repeats = experiment.get("repeats", 1)
            if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 100:
                raise BridgeError("paired request repeats must be an integer between 1 and 100")
            slug = bridge.slug(str(experiment.get("id") or f"request-{index}"))
            source_plan_path = (out_dir / f"pair-plan-{index}-{slug}.json").resolve()
            bridge.write_json_atomic(source_plan_path, pair_plan)
            for repeat in range(repeats):
                plan_path = source_plan_path
                if repeats > 1:
                    repeated_plan = copy.deepcopy(pair_plan)
                    metadata = repeated_plan.setdefault("metadata", {})
                    source_id = str(metadata.get("experiment_id") or slug)
                    metadata.update(
                        experiment_id=f"{source_id}-r{repeat + 1:02d}",
                        repeat=repeat + 1,
                        source_experiment_id=source_id,
                    )
                    plan_path = (out_dir / f"pair-plan-{index}-{slug}-r{repeat + 1:02d}.json").resolve()
                    bridge.write_json_atomic(plan_path, repeated_plan)
                out = out_dir / f"{slug}_middle_pair_r{repeat + 1}"
                command = [
                    "python3",
                    "tools/r2s_pair_run.py",
                    "--plan",
                    str(plan_path),
                    "--out",
                    str(out),
                    "--effort-raw",
                    str(int(effort_raw)),
                ]
                lines.extend(
                    [
                        "# Paired payload: validate and dry-run locally before execution.",
                        " ".join(bridge.shlex.quote(part) for part in command),
                        "",
                    ]
                )
                plan.append(
                    {
                        "execution_mode": "paired",
                        "joints": list(joints),
                        "pair_plan": str(plan_path),
                        "repeat": repeat + 1,
                        "source_pair_plan": str(source_plan_path),
                        "output": str(out),
                        "execute": "local pair runner validates bounds, controller and both joints before --execute",
                    }
                )
            continue
        profile = str(experiment.get("profile") or "step")
        center_deg = experiment.get("center_deg")
        amplitude_deg = experiment.get("amplitude_deg")
        center_rad = (
            float(center_deg) * math.pi / 180.0
            if center_deg is not None
            else (None if experiment.get("center_rad") is None else float(experiment["center_rad"]))
        )
        amplitude_rad = (
            float(amplitude_deg) * math.pi / 180.0
            if amplitude_deg is not None
            else (float(experiment.get("amplitude_rad") or 0.02))
        )
        speed = experiment.get("speed_cap_deg_s")
        speed_rad = (
            float(speed) * math.pi / 180.0
            if speed is not None
            else float(experiment.get("speed_limit_rad_s") or 0.2618)
        )
        duration = float(experiment.get("duration_s") or 8.0)
        rate = float(experiment.get("rate_hz") or 100.0)
        frequency = float(experiment.get("frequency_hz") or 0.1)
        end_frequency = float(experiment.get("end_frequency_hz") or 1.0)
        repeats = int(experiment.get("repeats") or 1)
        for repeat in range(repeats):
            for joint in joints:
                slug = bridge.slug(str(experiment.get("id") or f"request-{index}"))
                out = out_dir / f"{slug}_{joint}_r{repeat + 1}"
                command = [
                    "python3",
                    "tools/r2s_run.py",
                    "--joint",
                    str(joint),
                    "--profile",
                    profile,
                    "--out",
                    str(out),
                    "--duration-s",
                    repr(duration),
                    "--rate-hz",
                    repr(rate),
                    "--speed-limit-rad-s",
                    repr(speed_rad),
                    "--frequency-hz",
                    repr(frequency),
                    "--end-frequency-hz",
                    repr(end_frequency),
                    "--effort-raw",
                    str(int(effort_raw)),
                ]
                if center_rad is not None:
                    command.extend(["--center-rad", repr(center_rad)])
                if amplitude_rad is not None:
                    command.extend(["--amplitude-rad", repr(amplitude_rad)])
                expression = " ".join(bridge.shlex.quote(part) for part in command)
                lines.append(f"# {experiment.get('purpose', '')}")
                lines.append(expression)
                lines.append("")
                plan.append(
                    {
                        "joint": joint,
                        "profile": profile,
                        "center_rad": center_rad,
                        "amplitude_rad": amplitude_rad,
                        "speed_limit_rad_s": speed_rad,
                        "duration_s": duration,
                        "frequency_hz": frequency,
                        "repeats": repeats,
                        "output": str(out),
                        "execute": "add --execute after hardware/port/safety checks",
                    }
                )
    script = out_dir / "run_plan.sh"
    bridge.write_text_atomic(script, "\n".join(lines), mode=0o755)
    record = {
        "schema_version": bridge.SCHEMA_VERSION,
        "source_request": str(request_path.resolve()),
        "created_at_utc": bridge.utc_now(),
        "plan": plan,
        "script": str(script),
        "note": (
            "Dry command plan only; review limits and run on hardware explicitly. "
            "For separate requests, directions/dwell_s describe intent and r2s_run profiles encode "
            "their stages internally. Paired requests use the frozen pair_plan points. "
            "Validate locally and use a new output directory for hardware execution."
        ),
    }
    bridge.write_json_atomic(out_dir / "plan.json", record)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="bridge.env path (default ~/.config/botyard-real2sim/bridge.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-config", help="write a local bridge.env (mode 600)")
    init.add_argument("--remote-host")
    init.add_argument("--remote-key", type=Path)
    init.add_argument("--remote-root", type=Path)
    init.add_argument("--local-experiment", type=Path)
    init.add_argument("--dsh-url")
    init.add_argument("--dsh-token", help="prefer R2S_REMOTE_DSH_TOKEN in the environment")
    init.add_argument("--dsh-session", help="fixed session id, or 'auto'")

    sub.add_parser("doctor", help="check SSH, rsync, remote directories, DSH token/session")

    build_self = sub.add_parser("build-selftest", help="build a synthetic transport self-test bundle")
    build_self.add_argument("bundle_id", nargs="?")
    build_self.add_argument("--replace", action="store_true")

    build_capture = sub.add_parser("build-capture", help="convert legacy host-aligned capture tables into a bundle")
    build_capture.add_argument("capture_dir", type=Path)
    build_capture.add_argument("--meta", type=Path, required=True, help="capture_meta.json with evidence and limits")
    build_capture.add_argument("--bundle-id")
    build_capture.add_argument(
        "--joints",
        nargs="+",
        default=None,
        help="joints to export (default: infer from commands.csv; one bundle per joint is allowed)",
    )
    build_capture.add_argument("--split", choices=("train", "validation", "test"), default="train")
    build_capture.add_argument("--intent", choices=("calibration", "transport_selftest"), default="calibration")
    build_capture.add_argument("--tolerance", action="append", default=[], metavar="CHANNEL=VALUE")
    build_capture.add_argument("--replace", action="store_true")
    build_capture.add_argument("--max-interpolation-gap-s", type=float, default=0.05)
    build_capture.add_argument("--no-unwrap-motor", action="store_true")
    build_capture.add_argument("--third-field-scale", type=float, default=0.01)
    build_capture.add_argument("--notes", default="")

    verify = sub.add_parser("verify", help="verify local bundle integrity and strict dataset readiness")
    verify.add_argument("bundle")
    verify.add_argument("--json", action="store_true")

    publish = sub.add_parser("publish", help="upload to remote staging, verify, mark ready, atomic rename")
    publish.add_argument("bundle")
    publish.add_argument("--replace", action="store_true")
    publish.add_argument("--skip-local-dataset", action="store_true")
    publish.add_argument("--no-sync-scripts", action="store_true")

    say = sub.add_parser("say", help="send an ad-hoc message to the remote DSH agent")
    say.add_argument("text")
    say.add_argument("--session", help="session id or 'auto'")
    say.add_argument("--no-wait", action="store_true")
    say.add_argument("--timeout-s", type=float, default=600.0)

    notify = sub.add_parser("notify", help="send the ready bundle to the remote DSH agent")
    notify.add_argument("bundle")
    notify.add_argument("--session", help="session id or 'auto' (default: config or latest main session)")
    notify.add_argument("--no-wait", action="store_true")
    notify.add_argument("--timeout-s", type=float, default=3600.0)

    status = sub.add_parser("status", help="show remote staging/ready/result state")
    status.add_argument("bundle", nargs="?")
    status.add_argument("--json", action="store_true")

    results = sub.add_parser("results", help="read or pull published remote results")
    result_sub = results.add_subparsers(dest="result_command", required=True)
    result_show = result_sub.add_parser("show")
    result_show.add_argument("bundle")
    result_show.add_argument("--json", action="store_true")
    result_pull = result_sub.add_parser("pull")
    result_pull.add_argument("bundle")
    result_pull.add_argument("--replace", action="store_true")

    watch = sub.add_parser("watch", help="poll until RESULT_READY.json appears")
    watch.add_argument("bundle")
    watch.add_argument("--timeout-s", type=float, default=3600.0)
    watch.add_argument("--interval-s", type=float, default=5.0)

    plan = sub.add_parser("plan-requests", help="turn next_data_request.json into single/pair dry-run commands")
    plan.add_argument("request", help="next_data_request.json path or bundle id with a pulled result")
    plan.add_argument("--out-dir", type=Path, required=True)
    plan.add_argument("--r2s-root", type=Path, default=Path("/home/botyard/real2sim/embedding_ws"))
    plan.add_argument("--effort-raw", type=int, default=0)

    return parser


def _run(args: argparse.Namespace, config: BridgeConfig) -> int:
    if args.command == "init-config":
        updated = BridgeConfig(
            remote_host=args.remote_host or config.remote_host,
            remote_key=(args.remote_key or config.remote_key).expanduser(),
            remote_root=(args.remote_root or config.remote_root).expanduser(),
            local_experiment=(args.local_experiment or config.local_experiment).expanduser(),
            dsh_url=args.dsh_url or config.dsh_url,
            dsh_token=args.dsh_token or config.dsh_token or os.environ.get("R2S_REMOTE_DSH_TOKEN"),
            dsh_session=args.dsh_session or config.dsh_session,
        )
        path = save_config(updated)
        print(f"wrote {path} (mode 600)")
        return 0
    if args.command == "doctor":
        return _doctor(config)
    if args.command == "build-selftest":
        report = make_selftest_bundle(config.local_experiment, bundle_id=args.bundle_id, replace=args.replace)
        dataset = validate_dataset_bundle(report["bundle_dir"])
        _print_json(
            {
                "bundle_dir": report["bundle_dir"],
                "bundle_id": Path(report["bundle_dir"]).name,
                "integrity_ok": report["ok"],
                "dataset_status": dataset.get("status"),
                "dataset_errors": dataset.get("errors"),
                "next": f"publish {Path(report['bundle_dir']).name}",
            }
        )
        return 0 if dataset.get("data_ready") else 2
    if args.command == "build-capture":
        tolerances = _tolerances(args)
        bundle_id = args.bundle_id or auto_bundle_id(Path(args.capture_dir).name)
        report = export_capture(
            args.capture_dir,
            meta_path=args.meta,
            out_root=config.local_experiment,
            bundle_id=bundle_id,
            joints=None if args.joints is None else tuple(args.joints),
            split=args.split,
            intent=args.intent,
            tolerances=tolerances,
            replace=args.replace,
            max_interpolation_gap_s=args.max_interpolation_gap_s,
            unwrap_motor=not args.no_unwrap_motor,
            third_field_scale_n_per_unit=args.third_field_scale,
            notes=args.notes,
        )
        dataset = validate_dataset_bundle(report["bundle_dir"])
        _print_json(
            {
                "bundle_dir": report["bundle_dir"],
                "bundle_id": bundle_id,
                "integrity_ok": report["ok"],
                "dataset_status": dataset.get("status"),
                "dataset_errors": dataset.get("errors"),
                "blocking_reasons": dataset.get("blocking_reasons"),
                "metrics": dataset.get("metrics"),
            }
        )
        return 0 if dataset.get("data_ready") else 2
    if args.command == "verify":
        bundle_dir = _resolve_bundle(args.bundle, config)
        integrity = verify_bundle_integrity(bundle_dir)
        dataset = validate_dataset_bundle(bundle_dir)
        payload = {"integrity": integrity, "dataset": dataset}
        if args.json:
            _print_json(payload)
        else:
            print(f"bundle: {bundle_dir}")
            print(f"integrity: {'OK' if integrity['ok'] else 'FAILED'} ({len(integrity.get('errors') or [])} errors)")
            print(
                f"dataset: {dataset.get('status')} data_ready={dataset.get('data_ready')} "
                f"fit_ready={dataset.get('fit_ready')} evaluation_ready={dataset.get('evaluation_ready')}"
            )
            for item in dataset.get("errors") or []:
                print(f"  error: {item.get('code')}: {item.get('message')}")
            for item in dataset.get("blocking_reasons") or []:
                print(f"  blocker: {item.get('code')}: {item.get('message')}")
        return 0 if integrity["ok"] and dataset.get("data_ready") else 2
    if args.command == "publish":
        bundle_dir = _resolve_bundle(args.bundle, config)
        state = _publish(
            config,
            bundle_dir,
            replace=args.replace,
            skip_local_dataset=args.skip_local_dataset,
            sync_scripts=not args.no_sync_scripts,
        )
        _print_json(
            {
                "bundle_id": state["bundle_id"],
                "remote_bundle": state["remote_bundle"],
                "published_at_utc": state["published_at_utc"],
                "ready": state["remote_receipt"].get("ok"),
                "next": f"notify {state['bundle_id']}",
            }
        )
        return 0
    if args.command == "say":
        outcome = _say(
            config,
            args.text,
            session=args.session,
            wait=not args.no_wait,
            timeout_s=args.timeout_s,
        )
        print(f"session: {outcome['session_id']}")
        print(outcome["final_text"] or "prompt accepted (no final text captured yet)")
        return 0
    if args.command == "notify":
        bundle_id = ensure_bundle_id(args.bundle)
        outcome = _notify(
            config,
            bundle_id,
            session=args.session,
            wait=not args.no_wait,
            timeout_s=args.timeout_s,
        )
        print(f"session: {outcome['session_id']}")
        if outcome["final_text"]:
            print("\nremote final message:\n" + outcome["final_text"])
        else:
            print("prompt accepted (no final text captured yet)")
        return 0
    if args.command == "status":
        _status(config, args.bundle, as_json=args.json)
        return 0
    if args.command == "results":
        bundle_id = ensure_bundle_id(args.bundle)
        if args.result_command == "show":
            _result_show(config, bundle_id, raw=args.json)
            return 0
        target = _result_pull(config, bundle_id, replace=args.replace)
        print(f"pulled result to {target}")
        return 0
    if args.command == "watch":
        bundle_id = ensure_bundle_id(args.bundle)
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            probe = bridge.run_ssh(
                config,
                f"test -f {bridge.shlex.quote(_remote_result_path(config, bundle_id) + '/RESULT_READY.json')} "
                "&& echo ready || echo pending",
                check=False,
            )
            state = probe.stdout.strip()
            print(f"[{bridge.utc_now()}] {state}")
            if state == "ready":
                _result_show(config, bundle_id, raw=False)
                return 0
            time.sleep(max(1.0, args.interval_s))
        raise BridgeError(f"timed out waiting for result {bundle_id}")
    if args.command == "plan-requests":
        request = Path(args.request)
        if not request.is_file():
            bundle_id = ensure_bundle_id(args.request)
            request = config.local_experiment / bundle_id / "results" / "next_data_request.json"
        record = _plan_requests(request, args.out_dir, r2s_root=args.r2s_root, effort_raw=args.effort_raw)
        _print_json(record)
        return 0
    raise BridgeError(f"unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        return _run(args, config)
    except BridgeError as error:
        print(f"botyard-bridge: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("botyard-bridge: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
