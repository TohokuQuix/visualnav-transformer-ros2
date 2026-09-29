#!/usr/bin/env bash
set -euo pipefail
R=/tmp/eatnav-quality-identifiability-gate/phase-c-coverage
P=/tmp/eatnav-quality-identifiability-gate/phase-c-remediation
COMMON='--manifest /tmp/eatnav-extended/output/manifest.json --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --coverage-config /tmp/eatnav-train/coverage-config.yaml --output /tmp/eatnav-quality-identifiability-gate/phase-c-coverage'
DOCKER='export PYTHONPATH=/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:$PYTHONPATH'
mkdir -p /tmp/eatnav-quality-identifiability-gate/phase-c-coverage
exec > >(tee -a /tmp/eatnav-quality-identifiability-gate/phase-c-coverage/runner.log) 2>&1
if ! docker exec isaac-lab-onix test -f "$R/freeze.done.json"; then
  printf '%s STAGE0 freeze\n' "$(date --iso-8601=seconds)"
  docker exec isaac-lab-onix bash -lc "$DOCKER; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_phase_c_coverage.py freeze $COMMON --checkpoint $P/reference/best.pt --prior-states $P/candidate-states.json --prior-thresholds $P/candidate-thresholds.json"
fi
docker cp isaac-lab-onix:$R/diagnostic-state-ids.json "$R/diagnostic-state-ids.json"
docker cp isaac-lab-onix:$R/source-provenance.json "$R/source-provenance.json"
docker cp isaac-lab-onix:$R/artifact-hashes.json "$R/artifact-hashes.json"
printf '%s STAGE1 teacher audit\n' "$(date --iso-8601=seconds)"
docker exec isaac-lab-onix bash -lc "$DOCKER; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_phase_c_coverage_stage1.py $COMMON"
docker cp isaac-lab-onix:$R/teacher-multimodality.json "$R/teacher-multimodality.json"
printf '%s STAGE2/3 trace and sampling\n' "$(date --iso-8601=seconds)"
docker exec isaac-lab-onix bash -lc "$DOCKER; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_phase_c_coverage_sampling.py $COMMON --checkpoint $P/reference/best.pt --prior-thresholds $P/candidate-thresholds.json"
docker cp isaac-lab-onix:$R/sampling-coverage-decision.json "$R/sampling-coverage-decision.json"
printf '%s STAGE4 structured conditional\n' "$(date --iso-8601=seconds)"
docker exec isaac-lab-onix bash -lc "$DOCKER; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_phase_c_coverage_structured.py $COMMON --checkpoint $P/reference/best.pt --prior-thresholds $P/candidate-thresholds.json"
docker cp isaac-lab-onix:$R/structured-coverage-decision.json "$R/structured-coverage-decision.json"
printf '%s CORE_COVERAGE_STAGES_COMPLETE\n' "$(date --iso-8601=seconds)"
