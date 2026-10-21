"""Standard-library contract tests for the Canonical JSONL adapter."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexstl_nuplan.canonical import (
    CanonicalFormatError,
    build_scenario_geometry,
    ego_state,
    group_frames_by_record_id,
    infer_frame_dt,
    load_canonical_jsonl,
    obstacle_rows,
    parse_frame,
)


def make_frame(
    *,
    record_id: str = "nuplan-test",
    timestamp_ns: int = 0,
    frame_index: int = 0,
    units: str = "SI",
    ego_x: float = 5.0,
    obstacles: list[dict] | None = None,
) -> dict:
    lane_points = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]
    return {
        "schema_version": 1,
        "contract": {
            "units": units,
            "coordinate_frame": "map_cartesian",
            "ego_reference": "rear_axle_center",
            "obstacle_reference": "body_center",
            "prediction_time": "relative_to_anchor",
        },
        "source": {
            "record_id": record_id,
            "timestamp_ns": timestamp_ns,
            "frame_index": frame_index,
        },
        "ego": {
            "x": ego_x,
            "y": 0.1,
            "heading_rad": 0.0,
            "speed_mps": 4.0,
            "current_lane_id": 7,
            "target_lane_id": 7,
            "target_speed_mps": 6.0,
            "horizon_s": 3.0,
        },
        "lanes": [
            {
                "id": 7,
                "left_id": -1,
                "right_id": -1,
                "center": lane_points,
                "left_boundary": [[x, 2.0] for x, _ in lane_points],
                "right_boundary": [[x, -2.0] for x, _ in lane_points],
            }
        ],
        "obstacles": obstacles or [],
        "vehicle_profile": "pacifica",
    }


class CanonicalAdapterTests(unittest.TestCase):
    def test_parse_and_group_orders_frames(self) -> None:
        later = parse_frame(make_frame(timestamp_ns=200_000_000, frame_index=2), 1)
        earlier = parse_frame(make_frame(timestamp_ns=0, frame_index=0), 2)
        grouped = group_frames_by_record_id([later, earlier])
        self.assertEqual([frame.frame_index for frame in grouped["nuplan-test"]], [0, 2])
        self.assertAlmostEqual(infer_frame_dt(grouped, fallback_dt=0.2), 0.2)

    def test_rejects_incompatible_units(self) -> None:
        with self.assertRaisesRegex(CanonicalFormatError, "contract.units"):
            parse_frame(make_frame(units="feet"))

    def test_geometry_starts_at_ego_projection_and_infers_lane_width(self) -> None:
        frame = parse_frame(make_frame(), 1)
        geometry = build_scenario_geometry(frame, default_corridor_width=4.0)
        self.assertAlmostEqual(geometry.reference_path[0][0], 5.0, places=6)
        self.assertAlmostEqual(geometry.reference_path[0][1], 0.0, places=6)
        self.assertAlmostEqual(geometry.corridor_width, 4.0, places=3)
        self.assertEqual(geometry.lane_id, "7")
        self.assertEqual(geometry.corridor_width_source, "median_center_to_boundaries")
        self.assertGreater(geometry.route_length_m, 30.0)

    def test_obstacle_rows_follow_cpp_nine_column_contract(self) -> None:
        obstacle = {
            "id": "vehicle-42",
            "x": 12.0,
            "y": 1.0,
            "heading_rad": 0.1,
            "speed_mps": 3.0,
            "length_m": 4.5,
            "width_m": 1.9,
            "prediction_modes": [{"mode": "cv", "trajectory": []}],
        }
        frame = parse_frame(make_frame(obstacles=[obstacle]), 1)
        first = obstacle_rows(frame)
        second = obstacle_rows(frame)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(first[0]), 9)
        self.assertEqual(first[0][0], second[0][0])
        self.assertEqual(first[0][5], 0.0)
        self.assertEqual(first[0][8], 0.0)

    def test_ego_state_uses_rear_axle_contract_and_zero_steering_fallback(self) -> None:
        frame = parse_frame(make_frame(), 1)
        self.assertEqual(ego_state(frame), [5.0, 0.1, 0.0, 4.0, 0.0])

    def test_loader_reports_malformed_json_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(CanonicalFormatError, "line 2"):
                load_canonical_jsonl(path)

    def test_rejects_duplicate_frame_indices(self) -> None:
        first = parse_frame(make_frame(timestamp_ns=0, frame_index=0), 1)
        duplicate = parse_frame(make_frame(timestamp_ns=200_000_000, frame_index=0), 2)
        with self.assertRaisesRegex(CanonicalFormatError, "duplicate frame_index"):
            group_frames_by_record_id([first, duplicate])

    def test_rejects_nonuniform_cadence(self) -> None:
        frames = [
            parse_frame(make_frame(timestamp_ns=0, frame_index=0), 1),
            parse_frame(make_frame(timestamp_ns=200_000_000, frame_index=2), 2),
            parse_frame(make_frame(timestamp_ns=700_000_000, frame_index=7), 3),
        ]
        grouped = group_frames_by_record_id(frames)
        with self.assertRaisesRegex(CanonicalFormatError, "non-uniform"):
            infer_frame_dt(grouped, fallback_dt=0.2)


if __name__ == "__main__":
    unittest.main()
