# nuPlan integration for Lex-STL FOM-IGO

This directory connects the existing C++ Lex-STL/FOM-IGO planner to the
STCP Canonical JSONL v1 interface. It deliberately does not import
CommonRoad or duplicate STCP's nuPlan-devkit scenario extraction code.

## Integration boundary

The data path is:

1. STCP's scenario exporter reads the local nuPlan database and maps, and
   exports Canonical JSONL v1.
2. The canonical adapter validates units, coordinate-frame semantics,
   timestamps, lane geometry, ego state, and obstacle dimensions.
3. The replay runner builds a route-relative reference path from the current
   lane, instantiates the compiled IGO planner, and runs receding-horizon replay
   over the selected anchors.
4. The outputs preserve data/configuration fingerprints, per-scenario outcomes,
   solve times, controls, and predicted state trajectories.

The ego advances using the first step returned by the planner. Other-agent
observations are refreshed from recorded nuPlan anchors. This is a useful
integration and regression gate, but it is **not** an interactive multi-agent
simulator: other vehicles do not react to the planned ego trajectory.

## Canonical v1 semantic boundary

Canonical JSONL v1 currently does not encode speed-limit, intersection-priority,
stop-sign, bus-stop, or emergency-vehicle semantics. By default the runner
removes rules that require these missing fields from a temporary effective
configuration and lists them in the report. Pass --strict-map-semantics to
fail instead of disabling them. Do not interpret this safe-subset run as a
full reproduction of the CommonRoad paper's rulebook.

The current C++ obstacle interface accepts a single obstacle state, speed, and
optional longitudinal acceleration. Canonical prediction modes are not
consumed by this interface; when acceleration is absent, the adapter uses zero
acceleration. Multimodal prediction integration is a separate future change,
not something this adapter claims to have completed.

## 1. Export nuPlan anchors with STCP

Run this in the STCP repository inside the existing nuPlan Docker environment.
Adjust mount paths to match the local Ubuntu host. For one-container testing,
mount this planner repository alongside STCP (for example, at /workspace/lexstl)
and keep STCP mounted at /workspace/stcp. Run the exporter from /workspace/stcp,
then build and run the planner from /workspace/lexstl. Keep the generated JSONL
under the mounted STCP directory so it survives a docker run --rm container exit.

~~~bash
export DATA_ROOT=/data/datasets/AD_Datasets/nuPlan_mini/nuplan-v1.1_mini
export MAP_ROOT=/data/datasets/AD_Datasets/nuPlan_mini/maps
export PYTHON_BIN=/opt/nuplan_env/bin/python

mkdir -p /workspace/stcp/benchmark_runs/lexstl_nuplan
"$PYTHON_BIN" adapters/nuplan/scenario_exporter.py \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --output /workspace/stcp/benchmark_runs/lexstl_nuplan/canonical.jsonl \
  --sidecar-meta /workspace/stcp/benchmark_runs/lexstl_nuplan/sidecar_meta.json \
  --prediction-mode cv \
  --limit-scenarios 10 \
  --max-anchors-per-scenario 20 \
  --subsample-interval 4
~~~

The exporter subsamples recorded iterations. The replay runner infers the
cadence from timestamps and adjusts the planner step count to preserve the
configured physical horizon. If the exporter skips irregularly spaced anchors,
the runner fails closed rather than pretending the cadence is uniform.

## 2. Build the pybind module with the same Python ABI

The extension must be built with the same Python interpreter that will run the
replay. The existing STCP image documents /opt/nuplan_env/bin/python; use the
actual interpreter mounted in the container.

Install any missing build dependencies in that environment. The target project
requires CMake, a C++17 compiler, Eigen, yaml-cpp, pagmo, and pybind11. For
example, on an Ubuntu container where apt packages are available:

~~~bash
sudo apt-get update
sudo apt-get install -y cmake g++ libeigen3-dev libyaml-cpp-dev libpagmo-dev

export PY=/opt/nuplan_env/bin/python
"$PY" -m pip install pybind11 PyYAML

cmake -S commonroad-evaluations \
  -B commonroad-evaluations/build-nuplan \
  -DPython_EXECUTABLE="$PY" \
  -DPython3_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  -DBUILD_FOM_IGO_TESTS=ON

cmake --build commonroad-evaluations/build-nuplan \
  --target lex_stl_planner_bindings --parallel 2
~~~

If the container lacks the pagmo development package or the Python-specific
pybind11 CMake package, install them in the same container before building.
Do not copy a shared library compiled for a different Python minor version.

## 3. Run the first real-data smoke test

From the STL-Lex-MPPI-Planner repository root, in the same container and Python
environment used for the build:

~~~bash
export PY=/opt/nuplan_env/bin/python

"$PY" nuplan-evaluations/run_nuplan_canonical.py \
  --input /workspace/stcp/benchmark_runs/lexstl_nuplan/canonical.jsonl \
  --config commonroad-evaluations/config/default_config.yaml \
  --output-dir /workspace/stcp/benchmark_runs/lexstl_nuplan/igo_smoke \
  --max-scenarios 5 \
  --max-frames-per-scenario 20
~~~

The first run should use a small number of scenarios and anchors. Increase the
limits only after checking the reports and confirming the effective cadence,
reference path, corridor width, active rulebook, and planner success rate.

Outputs:

- summary.md: concise human-readable gate result.
- summary.json: configuration/data hashes, timing, success counts, active and
  disabled rules, and limitations.
- scenarios.jsonl: one outcome record per input scenario.
- ticks.jsonl: per-anchor state, first control, full planned trajectory,
  Lex-STL cost, considered obstacle IDs, and timing.
- effective_config.yaml: the exact config passed to the C++ planner.

A non-zero process exit means no planner tick succeeded. The reports are still
written when scenario setup or planning fails, when possible.

## 4. Adapter unit tests

These tests use only the Python standard library and synthetic frames; they do
not require nuPlan data or the compiled C++ extension.

~~~bash
python -m unittest discover -s nuplan-evaluations/tests -v
python -m compileall -q nuplan-evaluations
~~~

## Current limitations and next gates

1. Confirm a successful real-data run in the Ubuntu container.
2. Inspect failed records and verify that selected lanes, route lengths, and
   corridor widths match the visualized nuPlan map.
3. Add explicit canonical map semantics for speed limits, traffic controls,
   intersection priority, and emergency vehicles before re-enabling those rules.
4. Extend the C++ obstacle interface to consume aligned prediction modes, with
   explicit provenance and timestamp validation, before claiming multimodal
   prediction-aware planning.
5. Add an independent collision checker and common evaluation protocol before
   making safety or superiority claims.

This PR establishes a runnable adapter and replay path; it does not claim that
the real-data acceptance gates have already passed in this session.
