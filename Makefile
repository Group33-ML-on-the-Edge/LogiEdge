# LogiBridge / LogiEdge: common tasks. Everything runs in .work/ (a flat copy of the scripts and data; see tools/stage.sh).
# Run `make help` for the list.
WORK := .work
PY   := python3

.PHONY: help stage test collect clean stats dataset train variants-numpy results simulate demo-drift ota-cost \
        playbook-tests image ota-demo playbook reports tf-variants benchmark-tflite

help:
	@echo "make test            run all tests (stages .work/ first)"
	@echo "make stats           create training_stats.npy from 10 min of clean Normal data (Task C2)"
	@echo "make dataset         generate the labelled dataset (Task D1)"
	@echo "make train           train the model and apply the 88% gate (Task D1)"
	@echo "make variants-numpy  M1/M2/M3 in numpy and the comparison results (Task F1, F2)"
	@echo "make results         build benchmark_results.csv and pareto_chart.png (Task F2)"
	@echo "make demo-drift      PSI drift demo: clean -> combined -> clean (Task E1)"
	@echo "make ota-cost        bandwidth cost of the three OTA strategies (Task E3)"
	@echo "make reports         merge the individual reports into phase1/phase2/final PDFs"
	@echo "--- need Docker / TensorFlow / Ansible (not run by the author) ---"
	@echo "make tf-variants     build the real .tflite files (TensorFlow + tfmot)"
	@echo "make benchmark-tflite   five-metric benchmark of the .tflite files"
	@echo "make image           build the inference container; make ota-demo; make playbook"
	@echo "make collect         copy files generated in .work/ back into their folders"

stage:
	@bash tools/stage.sh

test: stage
	cd $(WORK) && LOGIBRIDGE_ROOT=$(CURDIR) $(PY) -m unittest discover -s . -p "test_*.py"

collect:
	@$(PY) tools/collect.py

clean:
	rm -rf $(WORK)

stats: stage
	cd $(WORK) && $(PY) preprocessing.py make-stats --out training_stats.npy

dataset: stage
	cd $(WORK) && $(PY) generate_dataset.py

train: stage
	cd $(WORK) && $(PY) train_model.py && $(PY) train_model.py --augment-onset --out model_weights_onset.npz --meta model_metadata_onset.json

variants-numpy: stage
	cd $(WORK) && $(PY) model_variants.py && $(PY) vibration_gap_check.py

results: stage
	cd $(WORK) && $(PY) benchmark.py --numpy --out benchmark_selfcheck.json --csv benchmark_selfcheck_raw.csv \
	  && $(PY) build_results_csv.py --out benchmark_results.csv && $(PY) make_pareto.py --csv benchmark_results.csv --out pareto_chart.png
	@$(MAKE) --no-print-directory collect

simulate: stage
	cd $(WORK) && $(PY) simulator.py --anomaly combined --anomaly-start 30 --dry-run --duration 60 --speed 30 --seed 1

demo-drift: stage
	cd $(WORK) && $(PY) drift_demo.py --json drift_demo_results.json
	@$(MAKE) --no-print-directory collect

ota-cost: stage
	cd $(WORK) && $(PY) ota_strategy_cost.py && $(PY) ota_bandwidth.py

playbook-tests: stage
	cd $(WORK) && $(PY) -m unittest test_playbook

reports:
	cd reports && $(PY) build_reports.py

tf-variants: stage
	cd $(WORK) && $(PY) build_variants_tf.py --outdir . --export model.tflite
	@$(MAKE) --no-print-directory collect

benchmark-tflite: stage
	cd $(WORK) && $(PY) benchmark.py --tflite M1=m1_fp32.tflite M2=m2_ptq_int8.tflite M3=m3_pruned_int8.tflite --tdp-watts 15

# Docker build context is .work/ (flat), where the Dockerfile's COPY lines are valid. Needs model.tflite (make tf-variants).
image: stage
	cd $(WORK) && docker build -t logiedge/inference:v1 .

ota-demo: stage
	cd $(WORK) && ./ota_demo.sh

playbook: stage
	cd $(WORK) && ./run_playbook_twice.sh
