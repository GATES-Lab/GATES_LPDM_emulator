import os
import subprocess
import time

def launch_slurm_job(run_id, seed, output_dir):
    log_dir = f"logs/repro_run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)

    job_name = f"repro_train_{run_id}"
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
    base_dir = "repro_check_outputs"
    seed = 42
    run1_out = os.path.join(base_dir, "run1")
    run2_out = os.path.join(base_dir, "run2")

    launch_slurm_job(run_id=1, seed=seed, output_dir=run1_out)
    launch_slurm_job(run_id=2, seed=seed, output_dir=run2_out)

    print("\n SLURM jobs submitted. Once they finish, you can compare the outputs with compare_runs.py")

if __name__ == "__main__":
    main()
