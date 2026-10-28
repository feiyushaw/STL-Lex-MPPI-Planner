#!/usr/bin/env python3
"""Export local nuPlan mini scenarios to this repository's Canonical JSONL v1 contract.

This script intentionally uses nuPlan-devkit directly. It has no runtime or
source dependency on STCP or any other planner repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lexstl_nuplan.canonical import (  # noqa: E402
    CanonicalFormatError,
    _project_to_polyline,
    group_frames_by_record_id,
    infer_frame_dt,
    parse_frame,
)


def _line_points(line: Any) -> list[list[float]]:
    """Extract [x, y] coordinates from nuPlan PolylineMapObject-like objects."""
    discrete = getattr(line, "discrete_path", None)
    if callable(discrete):
        discrete = discrete()
    if discrete is not None:
        points = []
        for state in discrete:
            if hasattr(state, "x") and hasattr(state, "y"):
                points.append([float(state.x), float(state.y)])
            elif isinstance(state, (list, tuple)) and len(state) >= 2:
                points.append([float(state[0]), float(state[1])])
        if len(points) >= 2:
            return points

    linestring = getattr(line, "linestring", None)
    if callable(linestring):
        linestring = linestring()
    coordinates = getattr(linestring, "coords", None)
    if coordinates is not None:
        points = [[float(point[0]), float(point[1])] for point in coordinates]
        if len(points) >= 2:
            return points
    return []


def _heading(points: Sequence[Sequence[float]], at_end: bool) -> float:
    first, second = (points[-2], points[-1]) if at_end else (points[0], points[1])
    return math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))


def _route_centerline(lane: Any, max_successors: int = 4) -> list[list[float]]:
    """Follow the least-turning outgoing lane chain to extend the local reference path."""
    points = _line_points(lane.baseline_path)
    if len(points) < 2:
        return []
    current = lane
    seen = {str(getattr(lane, "id", ""))}
    for _ in range(max_successors):
        try:
            outgoing = list(current.outgoing_edges)
        except Exception:
            break
        candidates = []
        end = points[-1]
        current_heading = _heading(points, at_end=True)
        for candidate in outgoing:
            candidate_id = str(getattr(candidate, "id", ""))
            if not candidate_id or candidate_id in seen:
                continue
            candidate_points = _line_points(candidate.baseline_path)
            if len(candidate_points) < 2:
                continue
            gap = math.hypot(candidate_points[0][0] - end[0], candidate_points[0][1] - end[1])
            if gap > 8.0:
                continue
            heading_error = abs(math.remainder(_heading(candidate_points, at_end=False) - current_heading, 2.0 * math.pi))
            candidates.append((heading_error + 0.02 * gap, candidate_id, candidate, candidate_points))
        if not candidates:
            break
        _, candidate_id, current, candidate_points = min(candidates, key=lambda item: item[0])
        seen.add(candidate_id)
        for point in candidate_points:
            if math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 1e-3:
                points.append(point)
    return points


def _lane_records(map_api: Any, ego_x: float, ego_y: float, ego_heading: float, radius_m: float) -> tuple[list[dict[str, Any]], str]:
    """Serialize nearby lane geometry and select the lane closest to the rear axle."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    nearby = map_api.get_proximal_map_objects(
        Point2D(ego_x, ego_y),
        radius_m,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )
    objects = []
    seen_ids: set[str] = set()
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        for lane in nearby.get(layer, []):
            lane_id = str(getattr(lane, "id", ""))
            if not lane_id or lane_id in seen_ids:
                continue
            seen_ids.add(lane_id)
            center = _line_points(lane.baseline_path)
            if len(center) < 2:
                continue
            _, _, _, _, distance, lane_heading = _project_to_polyline(center, ego_x, ego_y)
            heading_error = abs(math.remainder(lane_heading - ego_heading, 2.0 * math.pi))
            score = distance + 3.0 * heading_error
            objects.append((score, lane, center, distance))
    if not objects:
        raise CanonicalFormatError("nuPlan map query returned no lane centerlines near ego")

    objects.sort(key=lambda item: item[0])
    selected = objects[0][1]
    selected_id = str(selected.id)
    records: list[dict[str, Any]] = []
    for _, lane, center, _ in objects[:50]:
        record: dict[str, Any] = {
            "id": str(lane.id),
            "center": center,
            "left_boundary": _line_points(lane.left_boundary),
            "right_boundary": _line_points(lane.right_boundary),
        }
        if str(lane.id) == selected_id:
            route_center = _route_centerline(lane)
            if len(route_center) >= 2:
                record["route_center"] = route_center
        records.append(record)
    return records, selected_id


def _obstacle_records(tracked_objects: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, obj in enumerate(tracked_objects):
        try:
            box = obj.box
            center = box.center
            length, width = float(box.length), float(box.width)
            if not all(math.isfinite(v) for v in (length, width)) or length <= 0.0 or width <= 0.0:
                continue
            velocity = getattr(obj, "velocity", None)
            speed = math.hypot(float(velocity.x), float(velocity.y)) if velocity is not None else 0.0
            metadata = getattr(obj, "metadata", None)
            track_id = (
                getattr(obj, "track_token", None)
                or getattr(metadata, "track_token", None)
                or getattr(metadata, "track_id", None)
                or f"{getattr(getattr(obj, 'tracked_object_type', None), 'name', 'object')}-{index}"
            )
            rows.append({
                "id": str(track_id),
                "x": float(center.x),
                "y": float(center.y),
                "heading_rad": float(getattr(center, "heading", 0.0)),
                "speed_mps": max(0.0, speed),
                "length_m": length,
                "width_m": width,
            })
        except (AttributeError, TypeError, ValueError):
            continue
    return rows


def _scenario_frame(scenario: Any, iteration: int, record_id: str, map_radius_m: float) -> dict[str, Any]:
    ego = scenario.get_ego_state_at_iteration(iteration)
    rear_axle = ego.rear_axle
    velocity = ego.dynamic_car_state.rear_axle_velocity_2d
    speed = math.hypot(float(velocity.x), float(velocity.y))
    heading = float(rear_axle.heading)
    lanes, current_lane_id = _lane_records(
        scenario.map_api, float(rear_axle.x), float(rear_axle.y), heading, map_radius_m
    )
    tracked = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
    return {
        "schema_version": 1,
        "contract": {
            "units": "SI",
            "coordinate_frame": "map_cartesian",
            "ego_reference": "rear_axle_center",
            "obstacle_reference": "body_center",
            "prediction_time": "relative_to_anchor",
        },
        "source": {
            "record_id": record_id,
            "timestamp_ns": int(scenario.get_time_point(iteration).time_us) * 1000,
            "frame_index": int(iteration),
            "log_name": str(scenario.log_name),
            "scenario_type": str(scenario.scenario_type),
        },
        "ego": {
            "x": float(rear_axle.x),
            "y": float(rear_axle.y),
            "heading_rad": heading,
            "steering_angle_rad": float(ego.tire_steering_angle),
            "speed_mps": speed,
            "current_lane_id": current_lane_id,
            "target_lane_id": current_lane_id,
            "target_speed_mps": max(speed, 1.0),
            "horizon_s": 3.0,
        },
        "lanes": lanes,
        "obstacles": _obstacle_records(tracked),
        "vehicle_profile": "pacifica",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="nuPlan data root")
    parser.add_argument("--db-dir", type=Path, required=True, help="Directory containing mini split .db files")
    parser.add_argument("--map-root", type=Path, required=True, help="nuPlan map root")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-scenarios", type=int, default=10)
    parser.add_argument("--candidate-scenarios", type=int, default=60)
    parser.add_argument("--max-anchors-per-scenario", type=int, default=20)
    parser.add_argument("--subsample-ratio", type=float, default=0.25, help="0.25 samples 20 Hz logs at 5 Hz")
    parser.add_argument("--map-radius-m", type=float, default=80.0)
    args = parser.parse_args()
    if args.limit_scenarios < 1 or args.candidate_scenarios < args.limit_scenarios:
        parser.error("candidate-scenarios must be >= positive limit-scenarios")
    if args.max_anchors_per_scenario < 2:
        parser.error("max-anchors-per-scenario must be >= 2")
    if not 0.0 < args.subsample_ratio <= 1.0:
        parser.error("subsample-ratio must be within (0, 1]")
    if not args.db_dir.is_dir() or not list(args.db_dir.glob("*.db")):
        parser.error(f"no nuPlan .db files found in {args.db_dir}")
    if not args.map_root.is_dir():
        parser.error(f"nuPlan map root does not exist: {args.map_root}")

    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    except ImportError as exc:
        raise SystemExit(
            "nuPlan-devkit is unavailable in this Python environment; run with the "
            "Python interpreter from the nuPlan Docker environment."
        ) from exc

    builder = NuPlanScenarioBuilder(
        data_root=str(args.data_root),
        map_root=str(args.map_root),
        sensor_root=str(args.data_root),
        db_files=str(args.db_dir),
        map_version=args.map_version,
        include_cameras=False,
        max_workers=1,
        verbose=False,
        scenario_mapping=ScenarioMapping({}, args.subsample_ratio),
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=args.candidate_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
    )
    scenarios = builder.get_scenarios(scenario_filter, Sequential())
    if not scenarios:
        raise SystemExit("nuPlan scenario extraction returned zero scenarios")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    with args.output.open("w", encoding="utf-8") as output:
        for scenario in scenarios:
            if len(exported) >= args.limit_scenarios:
                break
            record_id = f"{scenario.log_name}:{scenario.token}"
            try:
                frame_count = min(args.max_anchors_per_scenario, scenario.get_number_of_iterations())
                if frame_count < 2:
                    raise CanonicalFormatError("scenario has fewer than two sampled anchors")
                frames = [
                    _scenario_frame(scenario, iteration, record_id, args.map_radius_m)
                    for iteration in range(frame_count)
                ]
                parsed = [parse_frame(frame, i + 1) for i, frame in enumerate(frames)]
                grouped = group_frames_by_record_id(parsed)
                cadence = infer_frame_dt(grouped, fallback_dt=0.2)
                if abs(cadence - 0.2) > 0.01:
                    raise CanonicalFormatError(
                        f"sampled cadence is {cadence:.4f}s, expected 0.2000s; "
                        "adjust --subsample-ratio or source timestamps"
                    )
                # Commit a scenario only after all frames satisfy the v1 contract.
                for frame in frames:
                    output.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
                exported.append({
                    "record_id": record_id,
                    "log_name": str(scenario.log_name),
                    "scenario_token": str(scenario.token),
                    "scenario_type": str(scenario.scenario_type),
                    "anchors": frame_count,
                    "cadence_s": cadence,
                    "map_name": str(scenario.map_api.map_name),
                })
                print(f"[exported] {record_id}: {frame_count} anchors at {cadence:.3f}s")
            except Exception as exc:
                skipped.append({
                    "record_id": record_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                print(f"[skipped] {record_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

    metadata = {
        "schema_version": 1,
        "producer": "STL-Lex-IGO-Planner/nuplan-evaluations/export_nuplan_mini.py",
        "data_root": str(args.data_root),
        "db_dir": str(args.db_dir),
        "map_root": str(args.map_root),
        "map_version": args.map_version,
        "subsample_ratio": args.subsample_ratio,
        "requested_scenarios": args.limit_scenarios,
        "exported_scenarios": len(exported),
        "exported_anchors": sum(item["anchors"] for item in exported),
        "skipped_scenarios": skipped,
        "scenarios": exported,
    }
    metadata["output_sha256"] = _sha256(args.output)
    sidecar_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Canonical JSONL: {args.output}")
    print(f"Sidecar metadata: {sidecar_path}")
    print(f"Exported {len(exported)}/{args.limit_scenarios} scenarios; skipped {len(skipped)} candidates")
    return 0 if len(exported) == args.limit_scenarios else 2


if __name__ == "__main__":
    raise SystemExit(main())
