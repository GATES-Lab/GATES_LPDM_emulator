import os, glob, subprocess, json, csv, time, shutil
import datetime

DIRECTORY = "/projects/b5s/data/met_archive/UM/SAHARA"
BASE_DIR  = "/home/b5s/jeffc.b5s/graphnet_LPDM_emulator"  # where CSV will be saved
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_CSV   = os.path.join(BASE_DIR, f"_nc_check_results_{timestamp}.csv")
TIMEOUT_S = 60  # per-file timeout

worker = shutil.which("python")
if worker is None:
    raise RuntimeError("Could not find 'python' executable in PATH")

files = sorted(glob.glob(os.path.join(DIRECTORY, "*.nc")))
print(f"Found {len(files)} NetCDF files in {DIRECTORY}\n")

results = []
ok = fail = timeout = crash = 0

for path in files:
    t0 = time.time()
    name = os.path.basename(path)
    try:
        proc = subprocess.run(
            [worker, "worker_check_nc.py", path],
            capture_output=True, text=True, timeout=TIMEOUT_S
        )
        elapsed = time.time() - t0
        stdout = (proc.stdout or "").strip()
        # Try to parse JSON if present
        payload = {}
        if stdout:
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                pass

        if proc.returncode == 0 and payload.get("status") == "ok":
            ok += 1
            print(f"[OK]   {name} — {payload.get('vars', 0)} vars, "
                  f"{payload.get('dims', 0)} dims, format={payload.get('format')} "
                  f"({elapsed:.2f}s)")
            results.append({
                "file": path, "status": "ok",
                "vars": payload.get("vars", ""),
                "dims": payload.get("dims", ""),
                "format": payload.get("format", ""),
                "error": "", "seconds": f"{elapsed:.2f}"
            })
        else:
            fail += 1
            msg = payload.get("error") or stdout or (proc.stderr or "").strip() or "unknown error"
            print(f"[FAIL] {name} — {msg} ({elapsed:.2f}s)")
            results.append({
                "file": path, "status": "fail",
                "vars": "", "dims": "", "format": "",
                "error": msg, "seconds": f"{elapsed:.2f}"
            })

    except subprocess.TimeoutExpired:
        timeout += 1
        print(f"[TIMEOUT] {name} — >{TIMEOUT_S}s")
        results.append({
            "file": path, "status": "timeout",
            "vars": "", "dims": "", "format": "",
            "error": f"timeout>{TIMEOUT_S}s", "seconds": f"{TIMEOUT_S}+"
        })
    except Exception as e:
        crash += 1
        print(f"[ERROR] {name} — {e}")
        results.append({
            "file": path, "status": "error",
            "vars": "", "dims": "", "format": "",
            "error": str(e), "seconds": ""
        })

# Write a CSV summary next to the data
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file","status","vars","dims","format","error","seconds"])
    w.writeheader()
    w.writerows(results)

print("\nSummary:")
print(f"  OK:      {ok}")
print(f"  FAIL:    {fail}")
print(f"  TIMEOUT: {timeout}")
print(f"  ERROR:   {crash}")
print(f"\nWrote: {OUT_CSV}")
