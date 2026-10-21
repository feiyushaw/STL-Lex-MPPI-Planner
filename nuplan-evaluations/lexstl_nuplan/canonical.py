"""Strict reader and geometry adapter for STCP Canonical JSONL v1."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, DefaultDict, Iterable, Mapping, Sequence


EXPECTED_CONTRACT = {
    "units": "SI",
    "coordinate_frame": "map_cartesian",
    "ego_reference": "rear_axle_center",
    "obstacle_reference": "body_center",
    "prediction_time": "relative_to_anchor",
}
RULES_REQUIRING_UNAVAILABLE_SEMANTICS = frozenset({
    "EMERGENCY_VEHICLE", "PRIORITY", "SPEED_LIMITS",
    "STOP_AT_STOP_SIGN", "SERVE_BUS_STOP",
})


class CanonicalFormatError(ValueError):
    """A Canonical JSONL frame violates the supported data contract."""


@dataclass(frozen=True)
class CanonicalFrame:
    line_number: int
    record_id: str
    timestamp_ns: int
    frame_index: int
    ego: Mapping[str, Any]
    lanes: Sequence[Mapping[str, Any]]
    obstacles: Sequence[Mapping[str, Any]]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ScenarioGeometry:
    reference_path: list[list[float]]
    corridor_width: float
    lane_id: str
    route_length_m: float
    lane_match_distance_m: float
    corridor_width_source: str


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalFormatError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CanonicalFormatError(f"{context} must be finite")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanonicalFormatError(f"{context} must be a non-negative integer")
    return value


def _id_key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _validated_points(value: Any, context: str, *, required: bool) -> list[list[float]]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or len(value) < 2:
        raise CanonicalFormatError(f"{context} must contain at least two [x, y] points")
    points: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise CanonicalFormatError(f"{context}[{index}] must be [x, y]")
        points.append([
            _finite_number(point[0], f"{context}[{index}].x"),
            _finite_number(point[1], f"{context}[{index}].y"),
        ])
    return points


def parse_frame(payload: Any, line_number: int = 1) -> CanonicalFrame:
    """Validate and wrap one Canonical JSONL v1 object."""
    prefix = f"line {line_number}"
    if not isinstance(payload, dict):
        raise CanonicalFormatError(f"{prefix}: top-level value must be an object")
    if isinstance(payload.get("schema_version"), bool) or payload.get("schema_version") != 1:
        raise CanonicalFormatError(f"{prefix}: only schema_version=1 is supported")

    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise CanonicalFormatError(f"{prefix}: missing contract object")
    for key, expected in EXPECTED_CONTRACT.items():
        if contract.get(key) != expected:
            raise CanonicalFormatError(
                f"{prefix}: contract.{key}={contract.get(key)!r}; expected {expected!r}"
            )

    source = payload.get("source")
    if not isinstance(source, dict):
        raise CanonicalFormatError(f"{prefix}: missing source object")
    record_id = source.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise CanonicalFormatError(f"{prefix}: source.record_id must be a non-empty string")
    timestamp_ns = _nonnegative_int(source.get("timestamp_ns"), f"{prefix}.source.timestamp_ns")
    frame_index = _nonnegative_int(source.get("frame_index"), f"{prefix}.source.frame_index")

    ego = payload.get("ego")
    if not isinstance(ego, dict):
        raise CanonicalFormatError(f"{prefix}: missing ego object")
    for key in ("x", "y", "heading_rad", "speed_mps"):
        _finite_number(ego.get(key), f"{prefix}.ego.{key}")
    if float(ego["speed_mps"]) < 0.0:
        raise CanonicalFormatError(f"{prefix}.ego.speed_mps cannot be negative")

    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise CanonicalFormatError(f"{prefix}: lanes must be a non-empty list")
    lane_ids: set[str] = set()
    for index, lane in enumerate(lanes):
        if not isinstance(lane, dict) or "id" not in lane:
            raise CanonicalFormatError(f"{prefix}.lanes[{index}] must be an object with an id")
        lane_id = _id_key(lane["id"])
        if lane_id in lane_ids:
            raise CanonicalFormatError(f"{prefix}: duplicate lane id {lane_id!r}")
        lane_ids.add(lane_id)
        _validated_points(lane.get("center"), f"{prefix}.lanes[{index}].center", required=True)
        for boundary in ("left_boundary", "right_boundary"):
            if lane.get(boundary) is not None:
                _validated_points(lane.get(boundary), f"{prefix}.lanes[{index}].{boundary}", required=True)

    obstacles = payload.get("obstacles", [])
    if not isinstance(obstacles, list):
        raise CanonicalFormatError(f"{prefix}: obstacles must be a list")
    for index, obstacle in enumerate(obstacles):
        context = f"{prefix}.obstacles[{index}]"
        if not isinstance(obstacle, dict) or "id" not in obstacle:
            raise CanonicalFormatError(f"{context} must be an object with an id")
        for key in ("x", "y", "heading_rad", "speed_mps", "length_m", "width_m"):
            _finite_number(obstacle.get(key), f"{context}.{key}")
        if float(obstacle["speed_mps"]) < 0.0:
            raise CanonicalFormatError(f"{context}.speed_mps cannot be negative")
        if float(obstacle["length_m"]) <= 0.0 or float(obstacle["width_m"]) <= 0.0:
            raise CanonicalFormatError(f"{context} dimensions must be positive")

    return CanonicalFrame(
        line_number, record_id, timestamp_ns, frame_index, ego, lanes, obstacles, payload
    )


def load_canonical_jsonl(path: str | Path) -> list[CanonicalFrame]:
    """Read non-empty JSONL lines and report malformed input with line numbers."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Canonical JSONL input does not exist: {input_path}")
    frames: list[CanonicalFrame] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanonicalFormatError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            frames.append(parse_frame(payload, line_number))
    if not frames:
        raise CanonicalFormatError(f"Canonical JSONL contains no frames: {input_path}")
    return frames


def group_frames_by_record_id(
    frames: Iterable[CanonicalFrame],
) -> dict[str, list[CanonicalFrame]]:
    """Group and order anchors deterministically; reject duplicate indices/timestamps."""
    grouped: DefaultDict[str, list[CanonicalFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.record_id].append(frame)
    result: dict[str, list[CanonicalFrame]] = {}
    for record_id in sorted(grouped):
        ordered = sorted(grouped[record_id], key=lambda item: (item.timestamp_ns, item.frame_index))
        previous: CanonicalFrame | None = None
        seen_indices: set[int] = set()
        for frame in ordered:
            if frame.frame_index in seen_indices:
                raise CanonicalFormatError(
                    f"record {record_id!r} contains duplicate frame_index={frame.frame_index}"
                )
            seen_indices.add(frame.frame_index)
            if previous is not None and frame.timestamp_ns <= previous.timestamp_ns:
                raise CanonicalFormatError(
                    f"record {record_id!r} timestamps must be strictly increasing"
                )
            previous = frame
        result[record_id] = ordered
    return result


def infer_frame_dt(
    grouped_frames: Mapping[str, Sequence[CanonicalFrame]],
    fallback_dt: float,
    *,
    relative_tolerance: float = 0.05,
) -> float:
    """Infer a uniform anchor cadence, or use the configured dt for single-frame inputs."""
    deltas: list[float] = []
    for record_id, frames in grouped_frames.items():
        for previous, current in zip(frames, frames[1:]):
            delta = (current.timestamp_ns - previous.timestamp_ns) / 1_000_000_000.0
            if not math.isfinite(delta) or delta <= 0.0:
                raise CanonicalFormatError(f"record {record_id!r} has a non-positive time interval")
            deltas.append(delta)
    if not deltas:
        if not math.isfinite(fallback_dt) or fallback_dt <= 0.0:
            raise CanonicalFormatError("fallback planner dt must be positive and finite")
        return float(fallback_dt)
    median_dt = statistics.median(deltas)
    tolerance = max(1e-4, abs(median_dt) * relative_tolerance)
    irregular = [delta for delta in deltas if abs(delta - median_dt) > tolerance]
    if irregular:
        examples = ", ".join(f"{delta:.6f}s" for delta in irregular[:3])
        raise CanonicalFormatError(
            f"non-uniform Canonical JSONL cadence: median={median_dt:.6f}s; "
            f"outlier intervals include {examples}. Export a uniform subsample or split the input."
        )
    return float(median_dt)


def _polyline_lengths(points: Sequence[Sequence[float]]) -> list[float]:
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.hypot(second[0] - first[0], second[1] - first[1]))
    return cumulative


def _project_to_polyline(
    points: Sequence[Sequence[float]], x: float, y: float
) -> tuple[list[float], float, int, float, float, float]:
    """Return projected point, arc length, segment index, fraction, distance, tangent heading."""
    if len(points) < 2:
        raise CanonicalFormatError("polyline projection requires at least two points")
    cumulative = _polyline_lengths(points)
    best: tuple[float, list[float], float, int, float, float] | None = None
    for index, (first, second) in enumerate(zip(points, points[1:])):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            continue
        fraction = max(0.0, min(1.0, ((x - first[0]) * dx + (y - first[1]) * dy) / length_sq))
        px, py = first[0] + fraction * dx, first[1] + fraction * dy
        distance_sq = (x - px) ** 2 + (y - py) ** 2
        segment_length = math.sqrt(length_sq)
        along = cumulative[index] + fraction * segment_length
        candidate = (distance_sq, [px, py], along, index, fraction, math.atan2(dy, dx))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise CanonicalFormatError("polyline has no non-degenerate segments")
    distance_sq, projected, along, index, fraction, heading = best
    return projected, along, index, fraction, math.sqrt(distance_sq), heading


def _distance_point_to_polyline(point: Sequence[float], line: Sequence[Sequence[float]]) -> float:
    return _project_to_polyline(line, float(point[0]), float(point[1]))[4]


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return _polyline_lengths(points)[-1]


def _lane_by_id(lanes: Sequence[Mapping[str, Any]], lane_id: Any) -> Mapping[str, Any] | None:
    if lane_id is None:
        return None
    wanted = _id_key(lane_id)
    return next((lane for lane in lanes if _id_key(lane.get("id")) == wanted), None)


def _select_lane(frame: CanonicalFrame) -> tuple[Mapping[str, Any], float]:
    ego_x, ego_y = float(frame.ego["x"]), float(frame.ego["y"])
    ego_heading = float(frame.ego["heading_rad"])
    for key in ("current_lane_id", "target_lane_id"):
        lane = _lane_by_id(frame.lanes, frame.ego.get(key))
        if lane is not None:
            center = _validated_points(lane.get("center"), f"lane {lane.get('id')}.center", required=True)
            distance = _project_to_polyline(center, ego_x, ego_y)[4]
            if distance <= 8.0:
                return lane, distance
    best: tuple[float, Mapping[str, Any], float] | None = None
    for lane in frame.lanes:
        center = _validated_points(lane.get("center"), f"lane {lane.get('id')}.center", required=True)
        _, _, _, _, distance, lane_heading = _project_to_polyline(center, ego_x, ego_y)
        heading_error = abs(math.remainder(lane_heading - ego_heading, 2.0 * math.pi))
        score = distance + 3.0 * heading_error
        if best is None or score < best[0]:
            best = (score, lane, distance)
    if best is None or best[2] > 8.0:
        raise CanonicalFormatError(
            f"record {frame.record_id!r} frame {frame.frame_index}: "
            "no lane centerline is within 8 m of the ego"
        )
    return best[1], best[2]


def build_scenario_geometry(
    frame: CanonicalFrame,
    default_corridor_width: float = 4.0,
    *,
    min_forward_distance_m: float = 10.0,
) -> ScenarioGeometry:
    """Build a route-relative reference path from the selected lane centerline."""
    if not math.isfinite(default_corridor_width) or default_corridor_width <= 0.0:
        raise CanonicalFormatError("default corridor width must be positive and finite")
    if not math.isfinite(min_forward_distance_m) or min_forward_distance_m <= 0.0:
        raise CanonicalFormatError("minimum forward distance must be positive and finite")
    lane, match_distance = _select_lane(frame)
    center = _validated_points(lane.get("center"), f"lane {lane.get('id')}.center", required=True)
    projected, _, segment_index, _, _, _ = _project_to_polyline(
        center, float(frame.ego["x"]), float(frame.ego["y"])
    )
    path: list[list[float]] = [projected]
    for point in center[segment_index + 1:]:
        if math.hypot(point[0] - path[-1][0], point[1] - path[-1][1]) > 1e-3:
            path.append([float(point[0]), float(point[1])])
    route_length = _polyline_length(path) if len(path) >= 2 else 0.0
    if route_length < min_forward_distance_m:
        raise CanonicalFormatError(
            f"record {frame.record_id!r} frame {frame.frame_index}: selected lane has only "
            f"{route_length:.2f} m of forward reference path; need at least "
            f"{min_forward_distance_m:.2f} m"
        )

    left = _validated_points(lane.get("left_boundary"), "lane.left_boundary", required=False)
    right = _validated_points(lane.get("right_boundary"), "lane.right_boundary", required=False)
    corridor_width = float(default_corridor_width)
    width_source = "configured_fallback"
    if left and right:
        widths: list[float] = []
        stride = max(1, len(center) // 40)
        for point in center[::stride]:
            width = _distance_point_to_polyline(point, left) + _distance_point_to_polyline(point, right)
            if math.isfinite(width) and 2.0 <= width <= 10.0:
                widths.append(width)
        if widths:
            candidate_width = float(statistics.median(widths))
            if 2.0 <= candidate_width <= 10.0:
                corridor_width, width_source = candidate_width, "median_center_to_boundaries"
    return ScenarioGeometry(
        reference_path=path,
        corridor_width=corridor_width,
        lane_id=_id_key(lane.get("id")),
        route_length_m=route_length,
        lane_match_distance_m=float(match_distance),
        corridor_width_source=width_source,
    )


def _stable_obstacle_id(value: Any) -> int:
    if not isinstance(value, bool):
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value).is_integer():
            number = int(value)
            if number >= 0:
                return number
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    stable_id = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return stable_id or 1


def obstacle_rows(frame: CanonicalFrame) -> list[list[float]]:
    """Convert obstacles to [id,x,y,heading,speed,acceleration,length,width,emergency]."""
    rows: list[list[float]] = []
    ids_seen: dict[int, str] = {}
    for obstacle in frame.obstacles:
        source_id = obstacle["id"]
        obstacle_id = _stable_obstacle_id(source_id)
        source_key = str(source_id)
        previous = ids_seen.get(obstacle_id)
        if previous is not None and previous != source_key:
            raise CanonicalFormatError(
                f"record {frame.record_id!r} frame {frame.frame_index}: obstacle ids "
                f"{previous!r} and {source_key!r} collide after integer mapping"
            )
        ids_seen[obstacle_id] = source_key
        acceleration = obstacle.get("acceleration_mps2", obstacle.get("acceleration", 0.0))
        acceleration_value = _finite_number(acceleration, f"obstacle {source_key}.acceleration_mps2")
        emergency = bool(obstacle.get("is_emergency_vehicle", obstacle.get("emergency_vehicle", False)))
        rows.append([
            float(obstacle_id), float(obstacle["x"]), float(obstacle["y"]),
            float(obstacle["heading_rad"]), float(obstacle["speed_mps"]),
            acceleration_value, float(obstacle["length_m"]), float(obstacle["width_m"]),
            1.0 if emergency else 0.0,
        ])
    return rows


def ego_state(frame: CanonicalFrame) -> list[float]:
    """Return the planner state [x, y, steering_angle, speed, heading]."""
    return [
        float(frame.ego["x"]), float(frame.ego["y"]),
        float(frame.ego.get("steering_angle_rad", 0.0)),
        float(frame.ego["speed_mps"]), float(frame.ego["heading_rad"]),
    ]
