#!/usr/bin/env bash
# Run logibridge_deploy.yml twice with no changes in between and check that the second run reports changed=0.
# Both runs are printed so they can be recorded. Extra arguments go to ansible-playbook, e.g.  ./run_playbook_twice.sh -K
#
# Before the first run (once):
#   pip install ansible docker && ansible-galaxy collection install community.docker
#   ./prepare_demo_registry.sh          # local registry with logiedge/inference:v1
#   python3 convert_to_tflite.py        # model.tflite next to this script
# NOT RUN where this was written (no Ansible or Docker there).

set -euo pipefail
cd "$(dirname "$0")"

command -v ansible-playbook >/dev/null || { echo "ansible-playbook not found: pip install ansible" >&2; exit 1; }
[ -f model.tflite ] || { echo "model.tflite not found: run python3 convert_to_tflite.py" >&2; exit 1; }
[ -f reference_dist.json ] || { echo "reference_dist.json not found: run python3 drift_monitor.py make-reference" >&2; exit 1; }

for run in 1 2; do
  printf '\n############ RUN %s ############\n' "$run"
  ansible-playbook -i inventory.ini logibridge_deploy.yml "$@" | tee "run${run}.log"
done

printf '\n############ RESULT ############\n'
echo "First run:";  grep -E ': ok=' run1.log || true
echo "Second run:"; grep -E ': ok=' run2.log || true
if grep -E ': ok=' run2.log | grep -q 'changed=0 '; then
  echo "PASS: the second run reported changed=0"
else
  echo "FAIL: the second run reported changes" >&2
  exit 1
fi
