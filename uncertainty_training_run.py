import os
import subprocess
import time

def launch_slurm_job(run_id, seed, output_dir):
    log_dir = f"logs/uncertainty_run_{run_id}_seed_{seed}"
    os.makedirs(log_dir, exist_ok=True)

    job_name = f"uncertainty_train_{run_id}"
    log_path = os.path.join(log_dir, f"{job_name}.out")
    script = "isambardai_slurm.sh"

    env = os.environ.copy()
    env["OUTPUT_DIR"] = output_dir
    env["SEED"] = str(seed)

    print(f"Launching SLURM job for run {run_id}...")
    subprocess.run(
        ["sbatch", "--job-name", job_name, "--output", log_path, script],
        env=env,
        check=True
    )

def main():
    base_dir = "uncertainty_runs"
    seeds = [11, 23, 42, 87, 135
             ]

    for i, seed in enumerate(seeds, start=1):
        run_dir = os.path.join(base_dir, f"run_{i}_seed{seed}")
        launch_slurm_job(run_id=i, seed=seed, output_dir=run_dir)

    print("\nSLURM jobs submitted. Once they finish, you can compare the outputs with uncertainty_calculate.py")


if __name__ == "__main__":
    main()
