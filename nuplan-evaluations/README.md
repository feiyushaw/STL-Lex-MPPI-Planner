# nuPlan integration for Lex-STL FOM-IGO

This directory contains a self-contained nuPlan mini exporter, Canonical JSONL
v1 reader, and recorded-observation replay runner. It has no source or runtime
dependency on STCP or another planner repository. Canonical JSONL is the data
contract between nuPlan extraction and this planner.

## Validation boundary

The exporter uses nuPlan-devkit directly to read local mini databases and maps.
The replay runner builds a lane-relative reference path, invokes the compiled
C++ IGO planner, and records scenario/tick outcomes, input/configuration hashes,
planned states, controls, costs, and timing.

This is recorded-observation replay: the ego advances using the planner's first
predicted step, while other-agent observations are refreshed from recorded
nuPlan anchors. It is not an interactive multi-agent simulator.

## 1. Export local nuPlan mini scenarios

Run inside the existing nuPlan Docker environment, using the same Python
interpreter that will load the compiled pybind module. Defaults below match the
local Ubuntu nuPlan layout; override paths if the mount points differ.

~~~bash
export PY=/opt/nuplan_env/bin/python
export NUPLAN_ROOT=/data/datasets/nuplan/dataset
export DB_DIR=$NUPLAN_ROOT/nuplan-v1.1_mini/splits/mini
export MAP_ROOT=$NUPLAN_ROOT/maps
export OUT=nuplan-evaluations/outputs/nuplan_mini_canonical.jsonl

"$PY" nuplan-evaluations/export_nuplan_mini.py \
  --data-root "$NUPLAN_ROOT" \
  --db-dir "$DB_DIR" \
  --map-root "$MAP_ROOT" \
  --map-version nuplan-maps-v1.0 \
  --output "$OUT" \
  --limit-scenarios 10 \
  --candidate-scenarios 60 \
  --max-anchors-per-scenario 20 \
  --subsample-ratio 0.25
~~~

The exporter uses a 0.25 subsample ratio to convert 20 Hz database records to
5 Hz (0.2 s) anchors. It validates the contract and cadence before writing each
scenario and emits a metadata sidecar with hashes, selected scenarios, and
skipped-candidate reasons. Where map connectivity allows, the selected lane's
reference path is continued through outgoing lane segments. The planner-side
geometry adapter still validates forward distance and extends the rear
projection domain when necessary.

## 2. Build the C++/pybind module with the same Python ABI

~~~bash
export PY=/opt/nuplan_env/bin/python
"$PY" -m pip install pybind11 pyyaml
cmake -S commonroad-evaluations \
  -B commonroad-evaluations/build-nuplan \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY" \
  -DPython3_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  -DBUILD_FOM_IGO_TESTS=ON
cmake --build commonroad-evaluations/build-nuplan --parallel 2
ctest --test-dir commonroad-evaluations/build-nuplan --output-on-failure
PYTHONPATH="$PWD/commonroad-evaluations" "$PY" -c \
  'from lex_stl_planner.lib.lex_stl_planner_bindings import Config, IGOPlanner, PlannerResult; print(Config, IGOPlanner, PlannerResult().failure_reason)'
~~~

Do not reuse a shared library built for another Python minor version. The
planner exposes standard C++ exception messages through PlannerResult.failure_reason
and writes them to stderr instead of silently erasing the cause.

## 3. Run the real-data replay gate

~~~bash
export PY=/opt/nuplan_env/bin/python
export OUT=nuplan-evaluations/outputs/nuplan_mini_verified
PYTHONPATH="$PWD/commonroad-evaluations:$PWD/nuplan-evaluations" \
  "$PY" nuplan-evaluations/run_nuplan_canonical.py \
  --input nuplan-evaluations/outputs/nuplan_mini_canonical.jsonl \
  --config commonroad-evaluations/config/default_config.yaml \
  --output-dir "$OUT" \
  --max-scenarios 10 \
  --max-frames-per-scenario 20 \
  --min-backward-distance-m 12 \
  --min-success-rate 0.90
~~~

The replay writes summary.md, summary.json, scenarios.jsonl, ticks.jsonl, and
effective_config.yaml. Failed ticks include the original Canonical frame,
transformed planner state, obstacle rows, schedule target, and C++ exception
diagnostic where available. The 90% success rate is an acceptance gate, not a
claim that collision safety or rule compliance has already been proven.

## Canonical v1 limitations

Canonical JSONL v1 does not currently encode speed limits, intersection
priority, stop signs, bus stops, or emergency-vehicle semantics. Rules requiring
those fields are disabled in the effective configuration and listed in the
report; --strict-map-semantics rejects that fallback. The C++ obstacle interface
consumes current obstacle state and speed, not multimodal prediction modes.
Missing obstacle acceleration defaults to zero. A separate collision checker
and common evaluation protocol are required before making safety or
planner-superiority claims.

## Adapter tests

~~~bash
python -m unittest discover -s nuplan-evaluations/tests -v
python -m compileall -q nuplan-evaluations
~~~
