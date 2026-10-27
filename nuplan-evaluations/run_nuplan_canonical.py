#!/usr/bin/env python3
"""Batch replay FOM-IGO on this repository's Canonical JSONL v1 frames.

This is a recorded-observation replay: the ego advances using the planner's
first predicted step while surrounding-agent observations are refreshed from
the recorded nuPlan anchors. It is not an interactive multi-agent simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from lexstl_nuplan.canonical import (
    RULES_REQUIRING_UNAVAILABLE_SEMANTICS,
    build_scenario_geometry,
    ego_state,
    group_frames_by_record_id,
    infer_frame_dt,
    load_canonical_jsonl,
    obstacle_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "commonroad-evaluations"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required by the planner configuration loader; install it in the "
            "same Python environment used to build the pybind module."
        ) from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"planner config is not a YAML mapping: {path}")
    return value


def _write_effective_config(
    source_config: Path,
    output_path: Path,
    effective_dt: float,
    verbose: bool,
    strict_map_semantics: bool,
) -> tuple[dict[str, Any], list[str], list[str], float, int]:
    import yaml

    config = _load_yaml(source_config)
    general = config.setdefault("planner", {}).setdefault("general", {})
    configured_dt = float(general.get("dt", 0.2))
    configured_steps = int(general.get("time_horizon", 15))
    if configured_dt <= 0.0 or configured_steps < 2:
        raise ValueError("planner.general.dt must be positive and time_horizon must be >= 2")
    requested_horizon_s = configured_dt * configured_steps
    effective_steps = max(2, int(round(requested_horizon_s / effective_dt)))
    general["dt"] = float(effective_dt)
    general["time_horizon"] = effective_steps
    config.setdefault("debugging", {})["verbose"] = bool(verbose)

    rules = config.setdefault("rules", {})
    rulebook = rules.get("rulebook_order", [])
    if not isinstance(rulebook, list):
        raise ValueError("rules.rulebook_order must be a YAML list")
    disabled = [
        str(rule) for rule in rulebook
        if str(rule) in RULES_REQUIRING_UNAVAILABLE_SEMANTICS
    ]
    if strict_map_semantics and disabled:
        raise ValueError(
            "Canonical JSONL v1 does not carry the semantics required by these active "
            f"rules: {', '.join(disabled)}. Enrich the canonical contract or use the "
            "default safe-subset mode for an integration smoke test."
        )
    rules["rulebook_order"] = [
        rule for rule in rulebook
        if str(rule) not in RULES_REQUIRING_UNAVAILABLE_SEMANTICS
    ]
    if not rules["rulebook_order"]:
        raise ValueError("effective rulebook is empty after removing unsupported map semantics")

    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return (
        config,
        [str(rule) for rule in rules["rulebook_order"]],
        disabled,
        requested_horizon_s,
        effective_steps,
    )


def _load_planner_bindings():
    if not PLANNER_ROOT.is_dir():
        raise RuntimeError(f"planner source directory not found: {PLANNER_ROOT}")
    sys.path.insert(0, str(PLANNER_ROOT))
    try:
        from lex_stl_planner.lib.lex_stl_planner_bindings import Config, IGOPlanner
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the compiled lex_stl_planner_bindings extension. Build it "
            "with the same Python interpreter used to run this script; see "
            "nuplan-evaluations/README.md."
        ) from exc
    return Config, IGOPlanner


def _state_to_list(state: Any) -> list[float]:
    return [float(value) for value in state]


def _scheduled_longitudinal_position(
    initial_s_m: float, reference_speed_mps: float, step: int, horizon_steps: int, dt: float
) -> float:
    """Absolute path-coordinate target, based at the ego's initial projected arc length."""
    return float(initial_s_m + reference_speed_mps * (step + horizon_steps - 1) * dt)


def _obb_intersects(center_a, heading_a, half_length_a, half_width_a, center_b, heading_b, half_length_b, half_width_b):
    """Separating-axis test for two oriented rectangles in Cartesian coordinates."""
    ca, sa = math.cos(heading_a), math.sin(heading_a)
    cb, sb = math.cos(heading_b), math.sin(heading_b)
    delta_x, delta_y = float(center_b[0]) - float(center_a[0]), float(center_b[1]) - float(center_a[1])
    for ax, ay in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
        distance = abs(delta_x * ax + delta_y * ay)
        ra = half_length_a * abs(ca * ax + sa * ay) + half_width_a * abs(-sa * ax + ca * ay)
        rb = half_length_b * abs(cb * ax + sb * ay) + half_width_b * abs(-sb * ax + cb * ay)
        if distance > ra + rb + 1e-9:
            return False
    return True


def _trajectory_collision_report(planned_states, obstacles, config, dt):
    """Check planned ego boxes against constant-heading/acceleration obstacle extrapolations."""
    ego_length, ego_width = max(0.1, float(config.ego.length)), max(0.1, float(config.ego.width))
    rear_axle_to_center = max(0.0, float(config.ego.wheel_base) * 0.5)
    pairs, checked = [], 0
    for plan_step, state in enumerate(planned_states):
        if len(state) < 5:
            continue
        x, y, heading = float(state[0]), float(state[1]), float(state[4])
        ego_center = (x + rear_axle_to_center * math.cos(heading), y + rear_axle_to_center * math.sin(heading))
        time_s = plan_step * dt
        for obstacle in obstacles:
            if len(obstacle) < 8:
                continue
            obstacle_id = int(float(obstacle[0]))
            ox, oy, oh = float(obstacle[1]), float(obstacle[2]), float(obstacle[3])
            speed, acceleration = float(obstacle[4]), float(obstacle[5])
            length, width = float(obstacle[6]), float(obstacle[7])
            if length <= 0.0 or width <= 0.0:
                continue
            distance = speed * time_s + 0.5 * acceleration * time_s * time_s
            obstacle_center = (ox + distance * math.cos(oh), oy + distance * math.sin(oh))
            checked += 1
            if _obb_intersects(ego_center, heading, ego_length * 0.5, ego_width * 0.5,
                               obstacle_center, oh, length * 0.5, width * 0.5):
                pairs.append({"plan_step": plan_step, "time_s": time_s, "obstacle_id": obstacle_id})
    return {
        "predicted_collision": bool(pairs),
        "predicted_collision_pair_count": len(pairs),
        "predicted_collision_pairs": pairs[:100],
        "collision_checked_pairs": checked,
        "collision_model": "oriented_boxes_constant_heading_constant_acceleration",
    }


def _run_scenario(
    record_id: str,
    frames: list,
    config: Any,
    planner_class: Any,
    *,
    max_frames: int,
    default_corridor_width: float,
    reference_speed_mps: float,
    min_forward_distance_m: float,
    min_backward_distance_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_frames = frames[:max_frames]
    first = selected_frames[0]
    geometry = build_scenario_geometry(
        first,
        default_corridor_width=default_corridor_width,
        min_forward_distance_m=min_forward_distance_m,
        min_backward_distance_m=min_backward_distance_m,
    )
    planner = planner_class(
        config,
        geometry.reference_path,
        geometry.corridor_width,
        [],  # speed limits: absent from Canonical JSONL v1
        [],  # stop signs: absent from Canonical JSONL v1
        [],  # intersection priority semantics: absent from Canonical JSONL v1
        [],  # bus stops: absent from Canonical JSONL v1
    )

    state = ego_state(first)
    dt = float(config.planner.general.dt)
    horizon_steps = int(config.planner.general.time_horizon)
    ticks: list[dict[str, Any]] = []
    solve_times_ms: list[float] = []
    wall_times_ms: list[float] = []
    failures = 0

    for step, frame in enumerate(selected_frames):
        scheduled_position = _scheduled_longitudinal_position(
            geometry.initial_s_m, reference_speed_mps, step, horizon_steps, dt
        )
        obstacles = obstacle_rows(frame)
        started = time.perf_counter()
        planner.updateDynamicData(state, obstacles, scheduled_position)
        result = planner.plan()
        wall_time_ms = (time.perf_counter() - started) * 1000.0
        solve_time_ms = max(0.0, float(result.solve_time) * 1000.0)
        wall_times_ms.append(wall_time_ms)
        solve_times_ms.append(solve_time_ms)

        tick: dict[str, Any] = {
            "record_id": record_id,
            "frame_index": frame.frame_index,
            "timestamp_ns": frame.timestamp_ns,
            "planner_success": bool(result.success),
            "cost": int(result.cost) if result.success else None,
            "solve_time_ms": solve_time_ms,
            "wall_time_ms": wall_time_ms,
            "initial_state": list(state),
            "observed_ego_state": ego_state(frame),
            "scheduled_longitudinal_position_m": scheduled_position,
            "input_obstacle_count": len(obstacles),
            "considered_obstacle_ids": list(planner.getConsideredObstacleIds()),
        }
        if not result.success:
            failures += 1
            tick["failure_reason"] = str(
                getattr(result, "failure_reason", "") or
                "planner returned success=false without a C++ exception diagnostic"
            )
            # Retain the exact failed input, transformed obstacle rows, and planner
            # schedule so this failure can be reproduced without the original database.
            tick["input_snapshot"] = {
                "canonical_frame": frame.raw,
                "planner_state": list(state),
                "obstacle_rows": obstacles,
                "scheduled_longitudinal_position_m": scheduled_position,
                "planner_dt_s": dt,
                "planner_horizon_steps": horizon_steps,
            }
            ticks.append(tick)
            break

        state_matrix = result.x
        control_matrix = result.u
        planned_states = state_matrix.T.tolist()
        planned_controls = control_matrix.T.tolist()
        collision_report = _trajectory_collision_report(planned_states, obstacles, config, dt)
        next_state = _state_to_list(state_matrix[:, 1].tolist())
        tick.update(collision_report)
        tick["first_control"] = _state_to_list(control_matrix[:, 0].tolist())
        tick["predicted_next_state"] = next_state
        tick["planned_states"] = [
            [float(value) for value in row] for row in planned_states
        ]
        tick["planned_controls"] = [
            [float(value) for value in row] for row in planned_controls
        ]
        ticks.append(tick)
        state = next_state

    scenario_summary = {
        "record_id": record_id,
        "frames_available": len(frames),
        "frames_requested": len(selected_frames),
        "ticks_attempted": len(ticks),
        "ticks_successful": sum(1 for tick in ticks if tick["planner_success"]),
        "ticks_failed": failures,
        "collision_checked_ticks": sum(1 for tick in ticks if "predicted_collision" in tick),
        "predicted_collision_ticks": sum(1 for tick in ticks if tick.get("predicted_collision") is True),
        "predicted_collision_tick_rate": (
            sum(1 for tick in ticks if tick.get("predicted_collision") is True)
            / sum(1 for tick in ticks if "predicted_collision" in tick)
            if any("predicted_collision" in tick for tick in ticks) else None
        ),
        "status": "failed" if failures else "completed",
        "lane_id": geometry.lane_id,
        "lane_match_distance_m": geometry.lane_match_distance_m,
        "initial_s_m": geometry.initial_s_m,
        "backward_buffer_m": geometry.backward_buffer_m,
        "forward_reference_distance_m": geometry.forward_reference_distance_m,
        "corridor_width_m": geometry.corridor_width,
        "corridor_width_source": geometry.corridor_width_source,
        "reference_path_length_m": geometry.route_length_m,
        "median_solve_time_ms": statistics.median(solve_times_ms) if solve_times_ms else None,
        "p95_solve_time_ms": _percentile(solve_times_ms, 0.95),
        "median_wall_time_ms": statistics.median(wall_times_ms) if wall_times_ms else None,
        "p95_wall_time_ms": _percentile(wall_times_ms, 0.95),
    }
    return scenario_summary, ticks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Canonical JSONL v1 file")
    parser.add_argument("--config", type=Path, default=PLANNER_ROOT / "config/default_config.yaml")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-scenarios", type=int, default=5)
    parser.add_argument("--max-frames-per-scenario", type=int, default=20)
    parser.add_argument("--dt", type=float, default=None, help="Override replay interval in seconds")
    parser.add_argument("--min-forward-distance-m", type=float, default=10.0)
    parser.add_argument("--min-backward-distance-m", type=float, default=12.0)
    parser.add_argument("--strict-map-semantics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--min-success-rate", type=float, default=0.0,
        help="Fail after writing reports if planner success rate is below this threshold (0..1)",
    )
    args = parser.parse_args()

    if args.max_scenarios <= 0 or args.max_frames_per_scenario <= 0:
        parser.error("--max-scenarios and --max-frames-per-scenario must be positive")
    if not math.isfinite(args.min_success_rate) or not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be finite and within [0, 1]")
    if args.dt is not None and (not math.isfinite(args.dt) or args.dt <= 0.0):
        parser.error("--dt must be positive and finite")
    if not math.isfinite(args.min_forward_distance_m) or args.min_forward_distance_m <= 0.0:
        parser.error("--min-forward-distance-m must be positive and finite")

    input_path, config_path = args.input.resolve(), args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_canonical_jsonl(input_path)
    grouped = group_frames_by_record_id(frames)
    base_config = _load_yaml(config_path)
    configured_dt = float(base_config.get("planner", {}).get("general", {}).get("dt", 0.2))
    observed_dt = infer_frame_dt(grouped, fallback_dt=configured_dt)
    effective_dt = float(args.dt if args.dt is not None else observed_dt)
    if effective_dt <= 0.0 or not math.isfinite(effective_dt):
        raise ValueError("effective planner dt must be positive and finite")

    effective_config_path = output_dir / "effective_config.yaml"
    effective_config, active_rules, disabled_rules, requested_horizon_s, effective_steps = (
        _write_effective_config(
            config_path, effective_config_path, effective_dt, args.verbose,
            args.strict_map_semantics,
        )
    )
    Config, IGOPlanner = _load_planner_bindings()
    config = Config(str(effective_config_path))

    default_corridor_width = float(effective_config.get("scenario", {}).get("corridor_width", 4.0))
    reference_speed_mps = float(effective_config.get("stl", {}).get("v_ref_sc", 6.333))
    if not math.isfinite(reference_speed_mps) or reference_speed_mps <= 0.0:
        raise ValueError("stl.v_ref_sc must be positive and finite")

    chosen_ids = sorted(grouped)[:args.max_scenarios]
    scenario_summaries: list[dict[str, Any]] = []
    all_ticks: list[dict[str, Any]] = []
    for record_id in chosen_ids:
        try:
            scenario_summary, ticks = _run_scenario(
                record_id, grouped[record_id], config, IGOPlanner,
                max_frames=args.max_frames_per_scenario,
                default_corridor_width=default_corridor_width,
                reference_speed_mps=reference_speed_mps,
                min_forward_distance_m=args.min_forward_distance_m,
                min_backward_distance_m=args.min_backward_distance_m,
            )
        except Exception as exc:
            scenario_summary = {
                "record_id": record_id,
                "frames_available": len(grouped[record_id]),
                "ticks_attempted": 0,
                "ticks_successful": 0,
                "ticks_failed": 0,
                "status": "setup_failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_input_snapshot": grouped[record_id][0].raw,
            }
            ticks = []
        scenario_summaries.append(scenario_summary)
        all_ticks.extend(ticks)
        print(
            f"[{scenario_summary['status']}] {record_id}: "
            f"{scenario_summary.get('ticks_successful', 0)}/"
            f"{scenario_summary.get('ticks_attempted', 0)} ticks"
        )

    ticks_attempted = sum(int(item.get("ticks_attempted", 0)) for item in scenario_summaries)
    ticks_successful = sum(int(item.get("ticks_successful", 0)) for item in scenario_summaries)
    ticks_failed = sum(int(item.get("ticks_failed", 0)) for item in scenario_summaries)
    solve_times = [float(t["solve_time_ms"]) for t in all_ticks if t.get("solve_time_ms") is not None]
    wall_times = [float(t["wall_time_ms"]) for t in all_ticks if t.get("wall_time_ms") is not None]
    collision_checked_ticks = sum(1 for tick in all_ticks if "predicted_collision" in tick)
    predicted_collision_ticks = sum(1 for tick in all_ticks if tick.get("predicted_collision") is True)
    summary = {
        "schema_version": 1,
        "evaluation": "lexstl_fom_igo_on_nuplan_canonical_jsonl_v1",
        "replay_mode": "ego_planner_first_step_with_recorded_other_agent_observations",
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "source_config_path": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "effective_config_path": str(effective_config_path),
        "effective_config_sha256": _sha256(effective_config_path),
        "records_in_input": len(grouped),
        "records_selected": len(chosen_ids),
        "effective_dt_s": effective_dt,
        "requested_horizon_s": requested_horizon_s,
        "effective_horizon_steps": effective_steps,
        "effective_horizon_s": effective_dt * effective_steps,
        "active_rulebook": active_rules,
        "disabled_rules_missing_canonical_semantics": disabled_rules,
        "ticks_attempted": ticks_attempted,
        "ticks_successful": ticks_successful,
        "ticks_failed": ticks_failed,
        "planning_success_rate": ticks_successful / ticks_attempted if ticks_attempted else 0.0,
        "median_solve_time_ms": statistics.median(solve_times) if solve_times else None,
        "p95_solve_time_ms": _percentile(solve_times, 0.95),
        "median_wall_time_ms": statistics.median(wall_times) if wall_times else None,
        "p95_wall_time_ms": _percentile(wall_times, 0.95),
        "collision_checked_ticks": collision_checked_ticks,
        "predicted_collision_ticks": predicted_collision_ticks,
        "predicted_collision_tick_rate": (
            predicted_collision_ticks / collision_checked_ticks if collision_checked_ticks else None
        ),
        "scenarios": scenario_summaries,
        "limitations": [
            "Canonical JSONL v1 does not carry speed-limit, intersection-priority, stop-sign, bus-stop, or emergency-vehicle semantics; rules requiring them are disabled and listed above.",
            "The current C++ obstacle interface receives current state, speed, and optional longitudinal acceleration only. Canonical prediction_modes are not consumed; absent acceleration is treated as zero (constant-velocity extrapolation).",
            "This is a recorded-observation replay, not an interactive multi-agent simulation: surrounding agents follow recorded anchors and do not react to the planned ego trajectory.",
            "The collision diagnostic uses oriented boxes and constant-heading/constant-acceleration obstacle extrapolation; it is not the official nuPlan collision scorer or a safety guarantee.",
            "This integration gate reports planner success, collision diagnostics, and runtime; it is not a benchmark-superiority claim.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "ticks.jsonl").open("w", encoding="utf-8") as handle:
        for tick in all_ticks:
            handle.write(json.dumps(tick, ensure_ascii=False) + "\n")
    with (output_dir / "scenarios.jsonl").open("w", encoding="utf-8") as handle:
        for item in scenario_summaries:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# Lex-STL FOM-IGO nuPlan Canonical Replay", "",
        f"- Input records: {len(grouped)}; selected: {len(chosen_ids)}",
        f"- Planner ticks: {ticks_successful}/{ticks_attempted} successful ({summary['planning_success_rate']:.1%})",
        f"- Effective cadence: {effective_dt:.4f} s; horizon: {effective_steps} steps ({effective_dt * effective_steps:.2f} s)",
        f"- Solve time: median={summary['median_solve_time_ms']} ms; p95={summary['p95_solve_time_ms']} ms",
        f"- Wall time: median={summary['median_wall_time_ms']} ms; p95={summary['p95_wall_time_ms']} ms",
        f"- Predicted collision ticks: {predicted_collision_ticks}/{collision_checked_ticks}; rate={summary['predicted_collision_tick_rate']}",
        f"- Disabled rules due to absent canonical semantics: {', '.join(disabled_rules) if disabled_rules else 'none'}",
        "", "## Per-scenario results", "",
        "| Record | Status | Successful ticks | Attempted ticks | Median solve (ms) | p95 solve (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in scenario_summaries:
        lines.append(
            f"| {item['record_id']} | {item['status']} | {item.get('ticks_successful', 0)} | "
            f"{item.get('ticks_attempted', 0)} | {item.get('median_solve_time_ms', '—')} | "
            f"{item.get('p95_solve_time_ms', '—')} |"
        )
    lines.extend(["", "## Interpretation boundary", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Summary: {output_dir / 'summary.md'}")
    print(f"Machine-readable report: {output_dir / 'summary.json'}")
    print(f"Per-tick trajectories: {output_dir / 'ticks.jsonl'}")
    if summary["planning_success_rate"] < args.min_success_rate:
        print(
            f"FAIL: planning success rate {summary['planning_success_rate']:.1%} "
            f"is below required {args.min_success_rate:.1%}",
            file=sys.stderr,
        )
        return 3
    return 0 if ticks_successful > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
