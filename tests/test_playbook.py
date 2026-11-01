"""Tests for logibridge_deploy.yml (Task E2). Run with:  python3 -m unittest -v test_playbook

Ansible is not needed. The tests check the playbook structure, render its Jinja templates, and run the two shell
tasks (4 and 7) against a fake `docker` command. They cannot prove what Ansible itself reports: run
./run_playbook_twice.sh for that.
"""

import os
import re
import stat
import subprocess
import tempfile
import unittest

import jinja2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
with open(os.path.join(HERE, "logibridge_deploy.yml")) as _f:
    PLAY = yaml.safe_load(_f)[0]
TASKS = PLAY["tasks"]
VARS = PLAY["vars"]

ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)
ENV.tests["changed"] = lambda r: bool(r.get("changed", False))        # Ansible's own "changed" test


def render(text, **extra):
    ctx = {"playbook_dir": HERE}                                       # Ansible's magic variable
    for k, v in VARS.items():                                          # resolve variables that refer to each other
        ctx[k] = ENV.from_string(v).render(**ctx) if isinstance(v, str) else v
    ctx.update(extra)
    return ENV.from_string(text).render(**ctx)


def module_of(task):
    return next(k for k in task if k not in ("name", "register", "when", "changed_when", "failed_when", "args"))


class TestStructure(unittest.TestCase):
    def test_one_play_with_exactly_seven_tasks(self):
        with open(os.path.join(HERE, "logibridge_deploy.yml")) as f:
            plays = yaml.safe_load(f)
        self.assertEqual(len(plays), 1)
        self.assertEqual(len(TASKS), 7)
        for forbidden in ("handlers", "pre_tasks", "post_tasks", "roles"):
            self.assertNotIn(forbidden, PLAY)
        self.assertFalse([t for t in TASKS if any(k.startswith(("include", "import")) for k in t)])

    def test_tasks_are_in_the_required_order(self):
        wanted = ["Create /opt/logibridge directory if not present",
                  "Copy new model file to /opt/logibridge/model.tflite",
                  "Copy updated reference_dist.json",
                  "Stop the running inference container if running",
                  "Pull updated container image from local Docker registry",
                  "Start inference container with environment variables",
                  "Wait 15 seconds and verify container is running"]
        for i, (task, text) in enumerate(zip(TASKS, wanted), start=1):
            self.assertEqual(task["name"], f"{i}. {text}")

    def test_modules(self):
        self.assertEqual([module_of(t) for t in TASKS],
                         ["ansible.builtin.file", "ansible.builtin.copy", "ansible.builtin.copy", "ansible.builtin.shell",
                          "community.docker.docker_image", "community.docker.docker_container", "ansible.builtin.shell"])

    def test_paths_and_names(self):
        self.assertEqual(VARS["install_dir"], "/opt/logibridge")
        self.assertEqual(render(TASKS[0]["ansible.builtin.file"]["path"]), "/opt/logibridge")
        self.assertEqual(TASKS[0]["ansible.builtin.file"]["state"], "directory")
        self.assertEqual(render(TASKS[1]["ansible.builtin.copy"]["dest"]), "/opt/logibridge/model.tflite")
        self.assertEqual(render(TASKS[2]["ansible.builtin.copy"]["dest"]), "/opt/logibridge/reference_dist.json")

    def test_pull_comes_from_the_local_registry(self):
        img = TASKS[4]["community.docker.docker_image"]
        self.assertEqual(img["source"], "pull")
        self.assertTrue(render(img["name"]).startswith("localhost:5000/"))

    def test_container_gets_the_environment_variables(self):
        c = TASKS[5]["community.docker.docker_container"]
        self.assertEqual(c["state"], "started")
        env = {k: render(v) for k, v in c["env"].items()}
        for key in ("MODEL_PATH", "REFERENCE_DIST_PATH", "TRUCK_ID", "MQTT_HOST", "MQTT_PORT", "DRIFT_SCORE"):
            self.assertIn(key, env)
        self.assertEqual(env["MODEL_PATH"], "/opt/logibridge/model.tflite")
        self.assertEqual(env["REFERENCE_DIST_PATH"], "/opt/logibridge/reference_dist.json")
        self.assertEqual(render(c["volumes"][0]), "/opt/logibridge:/opt/logibridge:ro")
        self.assertEqual(render(c["name"]), "logibridge-inference")

    def test_env_values_are_strings(self):
        for v in TASKS[5]["community.docker.docker_container"]["env"].values():
            self.assertIsInstance(v, str)

    def test_shell_tasks_declare_when_they_change_something(self):
        for i in (3, 6):
            self.assertIn("changed_when", TASKS[i])


class TestIdempotencyDesign(unittest.TestCase):
    def test_copy_results_are_registered_for_the_stop_guard(self):
        self.assertEqual(TASKS[1]["register"], "model_copy")
        self.assertEqual(TASKS[2]["register"], "reference_copy")

    def test_stop_runs_only_when_a_file_changed(self):
        cond = "{{ " + TASKS[3]["when"] + " }}"
        cases = [({"changed": False}, {"changed": False}, "False"),
                 ({"changed": True}, {"changed": False}, "True"),
                 ({"changed": False}, {"changed": True}, "True"),
                 ({"changed": True}, {"changed": True}, "True")]
        for model, ref, expected in cases:
            self.assertEqual(render(cond, model_copy=model, reference_copy=ref), expected)

    def test_stop_reports_changed_only_if_something_was_stopped(self):
        expr = "{{ " + TASKS[3]["changed_when"] + " }}"
        self.assertEqual(render(expr, stop_result={"stdout": "3f9c1a2b7d40\n"}), "True")
        self.assertEqual(render(expr, stop_result={"stdout": ""}), "False")
        self.assertEqual(render(expr, stop_result={"stdout": "\n"}), "False")

    def test_verify_task_never_reports_changed_and_fails_unless_running(self):
        self.assertIs(TASKS[6]["changed_when"], False)
        expr = "{{ " + TASKS[6]["failed_when"] + " }}"
        self.assertEqual(render(expr, verify_result={"stdout": "true\n"}), "False")
        self.assertEqual(render(expr, verify_result={"stdout": "false\n"}), "True")
        self.assertEqual(render(expr, verify_result={"stdout": ""}), "True")

    def test_image_is_only_force_pulled_on_request(self):
        self.assertIs(VARS["force_pull"], False)
        self.assertEqual(TASKS[4]["community.docker.docker_image"]["force_source"], "{{ force_pull }}")

    def test_first_deployment_would_create_the_container_in_task_6_not_task_4(self):
        # state=stopped would create a missing container; the playbook uses a shell stop instead
        self.assertNotIn("docker_container", TASKS[3])
        self.assertIn("docker stop", TASKS[3]["ansible.builtin.shell"])


FAKE_DOCKER = r"""#!/bin/bash
state_file="$FAKE_DOCKER_STATE"
state="$(cat "$state_file" 2>/dev/null || echo absent)"
case "$1" in
  ps)
    if [ "$state" = running ] && printf '%s ' "$@" | grep -q 'status=running'; then echo 3f9c1a2b7d40; fi ;;
  stop)
    echo "$2"; echo stopped > "$state_file" ;;
  inspect)
    if [ "$state" = running ]; then echo true; elif [ "$state" = stopped ]; then echo false; else echo "Error: No such object" >&2; exit 1; fi ;;
esac
"""


class TestShellTasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(self.tmp.name, "state")
        for name, body in (("docker", FAKE_DOCKER), ("sleep", "#!/bin/bash\nexit 0\n")):   # fake sleep: no real 15 s
            path = os.path.join(self.tmp.name, name)
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        self.env = dict(os.environ, PATH=self.tmp.name + os.pathsep + os.environ["PATH"], FAKE_DOCKER_STATE=self.state)

    def set_state(self, value):
        with open(self.state, "w") as f:
            f.write(value)

    def get_state(self):
        with open(self.state) as f:
            return f.read().strip()

    def run_task(self, index):
        script = render(TASKS[index]["ansible.builtin.shell"])
        return subprocess.run(["/bin/bash", "-c", script], env=self.env, capture_output=True, text=True)

    def test_stop_stops_a_running_container_and_reports_its_id(self):
        self.set_state("running")
        r = self.run_task(3)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "3f9c1a2b7d40")
        self.assertEqual(self.get_state(), "stopped")

    def test_stop_does_nothing_when_the_container_is_not_running(self):
        for state in ("stopped", "absent"):
            self.set_state(state)
            r = self.run_task(3)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "")
            self.assertEqual(self.get_state(), state)

    def test_stop_command_filters_on_the_exact_name_and_running_status(self):
        script = render(TASKS[3]["ansible.builtin.shell"])
        self.assertIn('name=^logibridge-inference$', script)
        self.assertIn("status=running", script)
        self.assertIn("pipefail", script)

    def test_verify_task_waits_15_seconds_then_checks_the_container(self):
        script = render(TASKS[6]["ansible.builtin.shell"])
        self.assertRegex(script, r"sleep 15")
        self.assertIn("{{.State.Running}}", script)                       # Docker's template survived Jinja
        self.assertIn("logibridge-inference", script)
        self.set_state("running")
        self.assertEqual(self.run_task(6).stdout.strip(), "true")
        self.set_state("stopped")
        self.assertEqual(self.run_task(6).stdout.strip(), "false")


class TestSupportingFiles(unittest.TestCase):
    def test_inventory_has_a_trucks_group(self):
        with open(os.path.join(HERE, "inventory.ini")) as f:
            text = f.read()
        self.assertIn("[trucks]", text)
        self.assertEqual(PLAY["hosts"], "trucks")

    def test_run_script_runs_twice_and_checks_changed_0(self):
        with open(os.path.join(HERE, "run_playbook_twice.sh")) as f:
            text = f.read()
        self.assertEqual(len(re.findall(r"for run in 1 2", text)), 1)
        self.assertIn("changed=0", text)
        self.assertIn("exit 1", text)
        for name in ("run_playbook_twice.sh", "prepare_demo_registry.sh"):
            r = subprocess.run(["bash", "-n", os.path.join(HERE, name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_container_reads_the_variables_the_playbook_sets(self):
        import inference_service as svc
        env = {k: render(v) for k, v in TASKS[5]["community.docker.docker_container"]["env"].items()}
        cfg = svc.Config.from_env(env)
        self.assertEqual(cfg.model_path, "/opt/logibridge/model.tflite")
        self.assertEqual(cfg.reference_dist_path, "/opt/logibridge/reference_dist.json")
        self.assertEqual((cfg.truck_id, cfg.mqtt_host, cfg.mqtt_port, cfg.drift_score),
                         ("T001", "localhost", 1883, "normal_confidence"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
