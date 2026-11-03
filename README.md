# LogiBridge / LogiEdge — Group 33
## Start-to-finish developer guide (BITS Machine Learning on Edge assignment)

**Read this README from top to bottom the first time.** It explains what you are building, what each command does, what output to expect, and how to verify the result. It is written for a developer who has never seen this project.

**What this project does:** Simulates sensors on a refrigerated truck, turns the readings into six features, trains a model to predict **Normal (0), Warning (1), Critical (2)**, runs the model locally in a Docker container using MQTT, detects data drift with PSI, and demonstrates updates with Docker layer caching and Ansible.

**Validated environment:** Windows 11 + WSL2 Ubuntu, Python 3.11 virtual environment, TensorFlow 2.14.1, tensorflow-model-optimization 0.8.0 installed, Docker 29.9.1, Ansible 2.20.1. The project was tested locally, **not on a physical Raspberry Pi or real trucks**. Measurements are from the host, and simulated data is used throughout.

> **Important accuracy note:** The working M3 model uses a NumPy-based structured-pruning procedure with a polynomial-decay schedule; it does **not** execute TFMOT pruning callbacks. Installing `tensorflow-model-optimization` does not mean the TFMOT pruning requirement was fulfilled. The dataset also has a known mismatch between simulation-mode Warning labels and temperature-based severity thresholds. Read **Known limitations** before claiming deployment readiness.

---

# Part A — Windows setup, launching WSL, and preparing Ubuntu

## Step 1. Install WSL 2 and Ubuntu on Windows (first-time setup only)

**These commands are Windows PowerShell commands. Do not run them in Ubuntu.**

1. Turn on your Windows computer.
2. Click **Start**, search for **PowerShell**, right-click it and choose **Run as administrator**.
3. To install WSL and Ubuntu, enter:

```powershell
wsl --install -d Ubuntu
```

4. Wait for installation. **Restart Windows if requested**.
5. On first launch, Ubuntu will ask you to create a **Linux username and password**. This is different from your Windows login. The password does not display while you type; this is normal.
6. Open PowerShell again and verify:

```powershell
wsl --status
wsl -l -v
```

**Expected:** Ubuntu is listed and `VERSION` is `2`. Its `STATE` may be `Stopped` or `Running`. If Ubuntu is listed as version 1, consult Microsoft's WSL migration guidance before continuing. If WSL is already installed and Ubuntu is version 2, **skip the installation command**.

> Windows 10/11 with a supported WSL configuration and hardware virtualization enabled is required. If installation reports a virtualization or Windows-feature error, resolve that Windows setup issue before proceeding.

## Step 2. Start WSL Ubuntu — do this at the beginning of EVERY project session

**Every time you restart Windows, open a new terminal, or return to the project later, start here:**

1. Click **Start** → open **PowerShell** or **Windows Terminal**. For everyday use, do **not** run as administrator.
2. At the Windows prompt, run:

```powershell
wsl
```

This starts the default WSL distribution. To select Ubuntu explicitly, use `wsl -d Ubuntu` instead.

**Before:**

```text
PS C:\Users\Developer>
```

**After:**

```text
developer@LAPTOP:~$
```

**Stop here if the prompt still begins with `PS`.** Linux commands such as `sudo`, `source`, `make`, and the commands in the remaining sections must be run **inside WSL Ubuntu**. You only run `wsl` **once per terminal session**, not before every command.

To exit Ubuntu, enter `exit`. The next time you open PowerShell, run `wsl` again.

### Daily quick start (after first-time setup)

Every subsequent day, you normally need only:

**Windows PowerShell:**

```powershell
wsl
```

**Inside Ubuntu — new developer's clone:**

```bash
cd ~/logibridge
```

**Inside Ubuntu — Group 33's existing D: drive checkout (alternative):**

```bash
cd /mnt/d/BITS/Classes/sem3/MLE/assignment/New_MLE_Assignment/New_MLE_Assignment/logibridge/logibridge
```

Then run the desired project command, for example `make test`. **Do not recreate `.venv` or reinstall packages every day.** If your clone has a different directory name, use that name instead of `~/logibridge`.

## Step 3. Check the Linux tools and Docker

In **Ubuntu**:

```bash
sudo apt update
sudo apt install -y git make ca-certificates

git --version
make --version
python3.11 --version
```

Git downloads the source; Make runs the project's build commands. If `python3.11` is missing, install a supported Python 3.11 interpreter **and its venv module** for your Ubuntu release before continuing. Do not substitute Python 3.14 for this project's validated TensorFlow environment.

Install **Docker Desktop on Windows** and enable **Settings → Resources → WSL Integration → Ubuntu**. Start Docker Desktop, then check in **Ubuntu**:

```bash
docker --version
docker compose version
docker info >/dev/null && echo "Docker access OK"
```

**Expected:** Docker version output and `Docker access OK`. If Docker access fails, check WSL integration and restart the Ubuntu terminal. You can postpone Docker installation until the Docker section, but it must be working before `make image`.

## Step 4. Clone the project or open the existing folder

**New developer — recommended:** Clone into Ubuntu's home directory for simpler permissions and better performance. In **Ubuntu**:

```bash
cd ~
git clone <PRIVATE_GITHUB_REPOSITORY_URL>
cd <CLONED_REPOSITORY_DIRECTORY>
```

Replace both placeholders with the real private GitHub URL and the actual cloned directory name.

**Existing Group 33 Windows checkout — alternative:** If the project is already on drive D:, use the WSL path:

```bash
cd /mnt/d/BITS/Classes/sem3/MLE/assignment/New_MLE_Assignment/New_MLE_Assignment/logibridge/logibridge
```

The Windows path `D:\BITS\...` becomes `/mnt/d/BITS/...` inside WSL. **Choose one of the two options; do not run both.**

Verify that you are at the repository root:

```bash
pwd
ls -la
```

**Expected:** `README.md`, `Makefile`, `requirements.txt`, `requirements-tf.txt`, `training/`, `inference/`, `deployment/`, `tests/` and other project folders. All later commands start here unless a step explicitly says `cd .work`.

## Step 5. Create the Python virtual environment in Ubuntu

The virtual environment belongs **inside the project directory** and must be created using Ubuntu's Python, not Windows Python:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
which python
```

**Expected:** Your terminal prompt begins with `(.venv)`; Python reports `3.11.x`, and `which python` ends in `.venv/bin/python`.

Whenever you open a **new terminal** for this project, return to the repository folder and run:

```bash
source .venv/bin/activate
```

If you get `No module named venv`, install the Python 3.11 venv support package for your Ubuntu release and rerun the creation command. The `.venv/` directory is local and **must not be committed to Git**.

## Step 6. Install Python dependencies

With `(.venv)` visible:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-tf.txt
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

**Expected in the validated environment:** TensorFlow `2.14.1`. The second requirements file installs TensorFlow-related packages for real `.tflite` model creation. Stop if imports fail; do not continue to model training with an incomplete environment.

## Step 7. Install Ansible for the deployment demonstration

In **Ubuntu** (Ansible can be installed later, before the Ansible section):

```bash
sudo apt install -y ansible
ansible-galaxy collection install community.docker
ansible --version
```

**Expected:** Ansible prints its version. The earlier successful run used Ansible core 2.20.1. If the Docker collection or playbook reports missing Python Docker SDK, install it for the interpreter Ansible actually uses. The deployment also requires Docker access and permissions for `/opt/logibridge`.

## Step 8. Understand the special `.work/` directory

The source is organized in folders, but scripts were developed to run together in one directory. The Makefile runs `tools/stage.sh` to copy files into a **temporary flat workspace** called `.work/`.

```bash
make stage
ls .work | head
```

- **Edit source files** in `training/`, `inference/`, etc., **never** inside `.work/`.
- `make stage` **recreates** `.work/`. Files you manually put there will be lost when staging runs again.
- `make collect` copies selected generated artifacts back into their source folders.
- `.work/` and `.venv/` are intentionally excluded from Git.

---

# Part B — Build the sensor data and train the model

## Step 9. See simulated sensor messages (Task C1)

```bash
make simulate
```

**What happens:** The simulator generates temperature, vibration and door-event messages without requiring MQTT (`--dry-run`). It uses the combined-fault mode in the Makefile.

**Check:** The terminal prints simulated sensor messages. Look for `truck/T001/temp`, `truck/T001/vib`, and door events. Normal temperature is around **4°C**, normal vibration around **0.45 g**; fault values change during the simulated anomaly.

For another dry-run mode:

```bash
python data_pipeline/simulator.py --anomaly temp_drift --dry-run --duration 60 --speed 30 --seed 1
```

**Interpretation:** These are *synthetic* sensor readings, not real truck measurements.

## Step 10. Generate normalisation statistics (Task C2)

```bash
make stats
make collect
ls -lh data_pipeline/training_stats.npy
```

**What happens:** Creates the frozen mean and standard deviation for the six model features using clean data.

**Check:** `training_stats.npy` exists. The model must use these same statistics during inference. **Do not recompute them from live faulty data**; that can hide anomalies.

## Step 11. Generate the labelled training dataset (Task D1)

```bash
make dataset
make collect
ls -lh training/data/
```

**What happens:** Produces labelled windows for Normal, Warning and Critical simulation modes.

**Check:** `training_dataset.npz` should exist under `training/data/`. The earlier validated dataset contained **291 windows** (117 Normal, 87 Warning, 87 Critical); exact counts can change if the simulator or generation settings change.

## Step 12. Train the neural network (Task D1)

```bash
make train
make collect
ls -lh training/models/
```

**What happens:** Trains the six-input, two-hidden-layer MLP (6 → 32 → 16 → 3), including a second run augmented with faults that begin after normal operation.

**Check:** Look for validation accuracy and model weight files such as `model_weights.npz` and `model_weights_onset.npz`. The assignment requires **at least 88% validation accuracy**. The earlier original model achieved **98.25%**, but a new run must be judged from its own output.

**Interpretation:** High accuracy on synthetic, overlapping windows is not proof of generalisation to real refrigerated trucks.

## Step 13. Create all three real TensorFlow Lite models (Task F1)

```bash
make tf-variants
ls -lh training/models/*.tflite inference/model.tflite
```

**What happens:**
- **M1 FP32** — baseline floating-point TFLite model.
- **M2 PTQ INT8** — post-training quantised model.
- **M3 pruned INT8** — smaller structured-pruned model, then quantised.

**Check:** All four files exist: `m1_fp32.tflite`, `m2_ptq_int8.tflite`, `m3_pruned_int8.tflite` in `training/models/`, and deployable `inference/model.tflite`.

Previously recorded sizes: **M1 5,100 bytes**, **M2 3,200 bytes**, **M3 2,768 bytes**. They may change on a new training run.

> **Important:** Do not rerun `make tf-variants` if your goal is to reproduce the *exact committed benchmark values*. Retraining or conversion can produce different model bytes. Use the committed models for a fast demonstration; regenerate models when testing the full build process.

---

# Part C — Test and benchmark the project

## Step 14. Run the automated tests

```bash
make test
echo "Test exit code: $?"
```

**Expected from the validated run:** `Ran 213 tests` followed by `OK`, with exit code `0`. A deliberately negative training test can print `FAILED: validation accuracy 33.3%` as **test fixture output**; the overall unittest summary and exit code determine success.

If tests fail, inspect the traceback rather than assuming the project passed because it previously passed on another machine.

## Step 15. Benchmark the three models (Tasks F2/F3)

```bash
make benchmark-tflite
make collect
ls -lh optimisation/results/
```

**What happens:** Runs inference timing and accuracy measurements on the machine executing the command. The benchmark is **host-based**, not a Raspberry Pi benchmark.

The previously verified run reported:

| Metric | M1 FP32 | M2 INT8 | M3 Pruned INT8 |
|---|---:|---:|---:|
| Mean latency (ms) | 0.0018 | 0.0164 | 0.0120 |
| P95 latency (ms) | 0.0020 | 0.0294 | 0.0130 |
| Model size (KB, decimal) | 5.10 | 3.20 | 2.77 |
| Held-out accuracy | 100% | 100% | 100% |
| Critical-class recall | 100% | 100% | 100% |
| Estimated energy (mJ/inference) | 0.00118 | 0.01131 | 0.00829 |

**How to read the results:** Lower latency and energy are better; smaller files reduce storage and model-transfer size; Critical recall must remain above **95%**. In this host run, M1 was fastest and M3 was smallest. The energy numbers are **estimates using a 15 W TDP assumption**, not power-meter measurements.

**Important Makefile behavior:** `make benchmark-tflite` runs the benchmark but does not itself call `make collect`. The subsequent `make collect` saves the supported results. The raw CSV may need to be copied manually if you want to retain it:

```bash
test -f .work/benchmark_results_raw.csv && cp .work/benchmark_results_raw.csv optimisation/results/
```

To regenerate the Pareto chart from the benchmark results:

```bash
cd .work
python build_results_csv.py --tflite benchmark_results.json --out benchmark_results.csv
python make_pareto.py --csv benchmark_results.csv --out pareto_chart.png
cd ..
make collect
```

**Check:** Open `optimisation/results/pareto_chart.png` and confirm that the chart corresponds to the CSV you just generated.

---

# Part D — Run live MQTT inference in Docker

## Step 16. Build the inference Docker image (Task D2)

```bash
make image
docker image ls logiedge/inference
```

**What happens:** Builds `logiedge/inference:v1` from the staged Dockerfile. It packages the Python inference service, frozen statistics and `model.tflite`.

**Check:** The image appears in `docker image ls`. The Dockerfile places `COPY model.tflite .` last so changing only the model can reuse earlier layers.

## Step 17. Start the Mosquitto broker

The repository includes a Compose file for the broker and inference service. For the simplest first run, start **both services together**, using the staged build context:

```bash
make stage
cd .work
docker compose up -d --build
docker compose ps
cd ..
```

**Expected:** `mosquitto` and `inference` are running. The broker publishes port **1883** on the host. Starting Compose from `.work/` is important because the repository Dockerfile expects the flattened source files.

> If port 1883 is already occupied or a previous `logibridge-inference` container is running, inspect `docker ps` and stop only the conflicting **project** containers you recognize. Do not remove unrelated Docker containers.

## Step 18. Publish a simulated fault to MQTT

Open a **second Ubuntu terminal**, go to the repository root and activate `.venv`:

```bash
source .venv/bin/activate
python data_pipeline/simulator.py \
  --anomaly combined \
  --host localhost --port 1883 \
  --truck-id T001 \
  --duration 180 --speed 10 \
  --anomaly-start 60 --seed 1 --verbose
```

**What happens:** Simulates normal operation followed by a combined temperature/vibration fault. The `--speed 10` flag accelerates simulated time.

## Step 19. Watch the predictions

In the first terminal:

```bash
cd .work
docker compose logs --tail 80 inference
cd ..
```

**Check:** Look for the model loading, MQTT connection, subscriptions to `truck/T001/temp` and `truck/T001/vib`, and prediction messages.

**Interpretation:**
- `Normal` / class `0`: model sees normal operating conditions.
- `Warning` / class `1`: model predicts temperature-drift warning.
- `Critical` / class `2`: model predicts a combined critical fault.
- Confidence is the model's score, **not a guarantee of correctness**.

The previous successful run showed Normal predictions before the anomaly and Critical predictions after it.

Optional live subscriber (requires `mosquitto-clients`):

```bash
sudo apt install -y mosquitto-clients
mosquitto_sub -h localhost -p 1883 -t 'logibridge/trucks/#' -v
```

When finished with this section, stop only the Compose stack you started:

```bash
cd .work
docker compose down
cd ..
```

---

# Part E — Drift detection (Task E1)

## Step 20. Run the PSI monitoring demonstration

```bash
make demo-drift
cat monitoring/results/drift_demo_results.json
```

**What happens:** Compares rolling Normal-class confidence scores with a clean reference distribution and simulates clean → combined anomaly → clean recovery.

**How to read PSI:**
- **PSI ≤ 0.25**: below the configured drift-alert threshold.
- **PSI > 0.25**: drift alert.
- **PSI < 0.10** after the fault: recovery threshold.

Previously verified: anomaly injected at simulated **20:00**, first alert at **21:30** with **PSI 0.387**, peak **2.217**, and recovery below 0.10 at **41:30**. This is a simulated-time demonstration; it does not prove live fleet recovery performance.

---

# Part F — Docker OTA update demonstration (Tasks D2/E3)

## Step 21. Prove that only the model layer changes

**Do not run `make ota-demo` directly after staging without preparing two distinct models.** It can otherwise compare identical files and incorrectly show 10/10 identical layers.

Run from the project root:

```bash
make stage
cp training/models/m1_fp32.tflite .work/model.tflite
cp training/models/m3_pruned_int8.tflite .work/model_v2.tflite

sha256sum .work/model.tflite .work/model_v2.tflite
```

**Check:** The two SHA-256 values must be different. Then run:

```bash
cd .work
./ota_demo.sh
cd ..
```

**What happens:** Builds v1 with M1, changes only `model.tflite` to M3, builds v2, compares Docker layer digests and estimates fleet transfer savings.

**Expected from the validated run:** **10 total layers, 9 shared, 1 new**. The earlier demonstration estimated **6,315.02 MB saved for 85 trucks**, approximately **₹631.50** at ₹0.10/MB. These are calculated transfer estimates, not observed cellular network usage.

**Note:** The script writes OTA logs and intermediate files inside `.work/`. Capture relevant evidence before another `make` command recreates that directory.

---

# Part G — Ansible deployment and idempotency (Task E2)

## Step 22. Build and push the image to the local demo registry

```bash
make image
make stage
cd .work
./prepare_demo_registry.sh
docker image ls localhost:5000/logiedge/inference
cd ..
```

**What happens:** Starts a local registry on port **5000**, tags the image and pushes it. This is a local demo registry, **not a remote fleet registry**.

**Check:** The image push completes. The registry container should be visible in `docker ps`.

## Step 23. Run the Ansible playbook twice

The playbook deploys to `localhost`, and the two-run script checks idempotency:

```bash
make stage
cd .work
./run_playbook_twice.sh -e use_become=false
cd ..
```

**Permission requirement:** `use_become=false` works only when the current user can access Docker and write to the deployment directory (`/opt/logibridge`). On a fresh machine, this may require an administrator to create the directory and grant appropriate ownership, or to use a supported `become` setup. Do **not** make `/opt` world-writable as a shortcut.

**Expected from the validated run:**
- First run: `ok=6 changed=1 skipped=1 failed=0`
- Second run: `ok=6 changed=0 skipped=1 failed=0`
- Script prints **PASS**.

**Interpretation:** `changed=0` on the second run proves that the playbook does not keep modifying an already-correct local deployment. It does not prove remote truck deployment.

Check the deployed container:

```bash
docker ps
docker logs --tail 40 logibridge-inference
```

**Check:** The inference service loads its model and connects to MQTT. This Ansible deployment uses host networking and therefore expects a broker reachable at `localhost:1883`. If you stopped the Compose broker in Step 19, start a broker before expecting MQTT connection:

```bash
docker compose -f inference/docker-compose.yml up -d mosquitto
```

Do not start a second broker if one already occupies port 1883.

---

# Part H — Review the assignment deliverables

## Step 24. Check the final evidence

| Assignment area | Where to find it | What to show or explain |
|---|---|---|
| A1/A2 | `scenario_architecture/` | Constraints and system architecture |
| B1/B2 | `hardware/` | Pi 5 recommendation and Roofline calculation |
| C1/C2/C3 | `data_pipeline/`, `training/` | Simulator, 30-second windows, six features, 3σ experiment |
| D1 | `training/models/`, `training/results/` | Training and validation |
| D2/D3 | `inference/`, `deployment/` | Docker, MQTT and ten-stage pipeline |
| E1 | `monitoring/results/` | PSI alert and recovery |
| E2 | Ansible terminal output | Second run `changed=0` |
| E3 | OTA terminal output | Layer reuse and bandwidth estimate |
| F1/F2/F3 | `training/models/`, `optimisation/results/` | M1/M2/M3, metrics, Pareto and recommendation |
| Final report | `reports/33_LogiEdge_Final.pdf` | Review for required diagrams, references and word count |
| Video | `demo/demo_video_link.txt` | Accessible 15–20-minute screen recording link |

**Do not assume all artifacts are ready just because a file exists.** Review the final PDF for required diagrams and accurate figures, and ensure the video URL is real and accessible.

## Step 25. Commit the files needed for another developer

**Commit:** source code, Makefile, README, both requirements files, scripts, tests, architecture/hardware documents, frozen statistics, model weights, all three `.tflite` variants, `inference/model.tflite`, drift reference/results, benchmark CSV/JSON/chart, report and demo link.

**Do not commit:** `.venv/`, `.work/`, `__pycache__/`, `*.bak`, local logs, credentials, private keys or `.env` files. The repository `.gitignore` should exclude these.

```bash
git status --short
git add README.md Makefile .gitignore SUBMISSION_CHECKLIST.md \
  requirements.txt requirements-tf.txt tools/ \
  scenario_architecture/ hardware/ data_pipeline/ training/ \
  inference/ monitoring/ optimisation/ deployment/ tests/ reports/ demo/
git diff --cached --stat
git diff --cached --check
```

Review the staged list and the modified tests before committing. If all files are correct:

```bash
git commit -m "Document and package reproducible LogiEdge Group 33 pipeline"
git push origin HEAD
```

A **fresh clone test** is recommended after committing: clone into a new directory and follow this README without copying `.venv/` or `.work/` from the original checkout.

---

# Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `python3.11: command not found` | Install a supported Python 3.11 interpreter with venv |
| TensorFlow import fails | Verify Python 3.11 and compatible versions in `requirements-tf.txt` |
| `No module named ...` | Activate `.venv` and install both requirements files |
| Docker permission denied | Start Docker Desktop / enable WSL integration / configure user access |
| Dockerfile cannot find `preprocessing.py` | Build using `make image` or staged `.work/`, not directly from `inference/` |
| MQTT connection refused | Broker not running, wrong hostname or port 1883 already in use |
| `container name already in use` | Inspect existing containers before removing a conflicting project container |
| `make` erased `.work/` changes | Expected behavior; edit source directories and use `make collect` |
| OTA shows 10 identical layers | Stage two **different** models as in Step 18 |
| Ansible fails pulling image | Prepare and push the local registry image in Step 19 |
| Ansible permission error | Configure Docker access and `/opt/logibridge` permissions |
| Second Ansible run has `changed>0` | Inspect changed tasks; idempotency is not yet verified |
| Benchmark differs from report | Host, runtime, model bytes or random training run differs |
| Report generator overwrites PDFs | Do not run `make reports` until you have backed up reviewed submission PDFs |

# Known limitations

1. Training and validation use **simulated data**. Results do not establish performance on real refrigerated trucks.
2. Benchmarks were collected under WSL, not on the proposed Raspberry Pi 5.
3. Estimated energy and OTA bandwidth savings are **calculations**, not physical meter or cellular-network readings.
4. M3 uses NumPy structured pruning, not the prescribed TFMOT pruning callback implementation.
5. Simulator-mode labels can disagree with temperature-based Critical severity. A production release needs severity-aligned labels and independent-trip testing.
6. The demo MQTT broker is not hardened for production. Add authentication, TLS, access controls and secure update/rollback before field use.

**Final success criterion:** A new developer can clone the repository, install dependencies, run the simulator, build and test models, benchmark variants, start Docker/MQTT inference, reproduce the PSI and OTA demonstrations, and verify Ansible `changed=0` on a second run—without copying any hidden files from the original developer's machine.
