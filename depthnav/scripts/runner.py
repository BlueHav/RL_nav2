import subprocess
import os
import re
import sys
import shlex
import yaml
import time
from copy import deepcopy
from depthnav.common import ExitCode


def update_nested_dict(d: dict, key: str, value):
    keys = key.split(".")
    sub_dict = d
    for k in keys[:-1]:
        # create sub-dicts if missing
        if k not in sub_dict:
            sub_dict[k] = {}
        sub_dict = sub_dict[k]
    sub_dict[keys[-1]] = value


def make_configs(experiment_dir, base_configs, config_keys, run_params):
    print("MAKING CONFIGS")
    # create config files for each run
    run_config_files = []
    run_names = []

    num_runs = len(run_params)
    for i, (run_name, run_values) in enumerate(run_params.items()):
        base_config = base_configs[i]

        run_config = deepcopy(base_config)
        for key, val in zip(config_keys, run_values):
            update_nested_dict(run_config, key, val)

        # write config to file
        run_config_file = os.path.join(experiment_dir, run_name + ".yaml")
        with open(run_config_file, "w") as file:
            yaml.dump(run_config, file, default_flow_style=None, sort_keys=False)

        run_names.append(run_name)
        run_config_files.append(run_config_file)
    return run_names, run_config_files


def extract_last_digits(s):
    # Find the last set of digits after the '_'
    match = re.search(r"_(\d+)(?=.pth)", s)
    return (
        int(match.group(1)) if match else float("inf")
    )  # Default to infinity if no digits found


def run_with_retries(command, max_retries=5):
    attempt = 0
    while attempt < max_retries:
        try:
            print(shlex.join(command))
            p = subprocess.Popen(
                command,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            p.wait()
            if p.returncode == ExitCode.SUCCESS.value:
                return True
            else:
                attempt += 1
                print(
                    f"Attempt {attempt}: {shlex.join(command)} failed with return code {p.returncode}"
                )
                time.sleep(1)
        except KeyboardInterrupt:
            print("Keyboard interrupt. Killing subprocess and exiting.")
            p.kill()
            sys.exit(ExitCode.KEYBOARD_INTERRUPT.value)
        except Exception as e:
            print(f"Error while running {shlex.join(command)}:\n{e}")
            attempt += 1
            time.sleep(1)
    return False


def run_experiment(
    script,
    experiment_dir,
    base_config_files,
    config_keys,
    run_params,
    policy_config_file=None,
    curriculum=False,
    eval_configs=None,
    eval_csvs=None,
    max_retries=5,
):
    """curriculum will use weights from last run_name in next run"""
    experiment_dir = os.path.abspath(experiment_dir)
    print("RUNNING EXPERIMENT")
    print(experiment_dir)

    if not os.path.exists(script):
        raise FileNotFoundError(f"Could not find training script: {script}")
    if eval_configs is None:
        eval_configs = []
    if eval_csvs is None:
        eval_csvs = []
    if eval_configs and len(eval_configs) != len(eval_csvs):
        raise ValueError("eval_configs and eval_csvs must have the same length")

    num_runs = len(run_params)
    if type(base_config_files) == str:
        base_config_files = [base_config_files for _ in range(num_runs)]
    assert len(base_config_files) == num_runs

    os.makedirs(experiment_dir, exist_ok=True)

    base_configs = []
    for base_config_file in base_config_files:
        if not os.path.exists(base_config_file):
            raise FileNotFoundError(f"Could not find base config file: {base_config_file}")
        with open(base_config_file, "r") as file:
            base_config = yaml.safe_load(file)

        if policy_config_file:
            if not os.path.exists(policy_config_file):
                raise FileNotFoundError(
                    f"Could not find policy config file: {policy_config_file}"
                )
            with open(policy_config_file, "r") as file:
                policy_config = yaml.safe_load(file)
            base_config.update(policy_config)

            if "update_env_kwargs" in policy_config:
                for k, v in policy_config["update_env_kwargs"].items():
                    base_config["env"][k] = v

        base_configs.append(base_config)

    if policy_config_file is None:
        updated_eval_configs = eval_configs
    else:
        updated_eval_configs = []
        for eval_config_file in eval_configs:
            if not os.path.exists(eval_config_file):
                raise FileNotFoundError(
                    f"Could not find eval config file: {eval_config_file}"
                )
            with open(eval_config_file, "r") as file:
                eval_config = yaml.safe_load(file)

            with open(policy_config_file, "r") as file:
                policy_config = yaml.safe_load(file)
            eval_config.update(policy_config)

            if "update_env_kwargs" in policy_config:
                for k, v in policy_config["update_env_kwargs"].items():
                    eval_config["env"][k] = v

            basename = os.path.basename(eval_config_file).split(".")[0]
            policy_name = os.path.basename(policy_config_file).split(".")[0]
            new_eval_path = os.path.join(
                experiment_dir, basename + "_" + policy_name + ".yaml"
            )
            with open(new_eval_path, "w") as file:
                yaml.dump(eval_config, file)
            updated_eval_configs.append(new_eval_path)

    run_names, run_config_files = make_configs(
        experiment_dir, base_configs, config_keys, run_params
    )

    # now run each config
    start_iter = 0
    for i, (run_name, run_config_file) in enumerate(zip(run_names, run_config_files)):
        print("=" * 80)
        command = [
            sys.executable,
            script,
            "--run_name",
            run_name,
            "--cfg_file",
            run_config_file,
            "--logging_root",
            experiment_dir,
        ]

        if curriculum and i > 0:
            last_run_name = run_names[i - 1]
            last_run_matches = [
                f
                for f in os.listdir(experiment_dir)
                if last_run_name in f and f.endswith(".pth") and "iteration" not in f
            ]
            last_run_matches.sort(key=extract_last_digits)
            if len(last_run_matches) == 0:
                raise IndexError(
                    f"Could not find weights file matching {last_run_name}"
                )
            last_run_pth = os.path.join(experiment_dir, last_run_matches[-1])
            command.extend(["--weight", last_run_pth, "--start_iter", str(start_iter)])

        if len(updated_eval_configs) > 0:
            command.append("--eval_configs")
            command.extend(updated_eval_configs)

            command.append("--eval_csvs")
            command.extend(eval_csvs)

        success = run_with_retries(command, max_retries)
        if not success:
            raise RuntimeError(
                f"Experiment stage '{run_name}' failed after {max_retries} retries"
            )
        # add number of iterations
        with open(run_config_file, "r") as file:
            run_config = yaml.safe_load(file)
        start_iter += int(run_config["train_bptt"]["iterations"])
