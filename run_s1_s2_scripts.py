
import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path, args=None):
    if args is None:
        args = []
    
    cmd = [sys.executable, str(script_path)] + args
    print(f"Running {script_path} with args {args}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"  ✓ Success")
            return True
        else:
            print(f"  ✗ Failed with return code {result.returncode}")
            print("STDERR:")
            print(result.stderr)
            print("STDOUT:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def main():
    scripts_dir = Path("scripts")
    
    # Ensure output directories exist
    for d in ["s1_level", "s1_delta", "s2_level", "s2_delta"]:
        (Path("outputs/seasonality") / d).mkdir(parents=True, exist_ok=True)

    scripts = [
        # S1 Level
        ("s1_overlay.py", ["--year", "2024", "--metric", "level", "--outdir", "outputs/seasonality/s1_level"]),
        ("s1_average_by_bdom.py", ["--year", "2024", "--metric", "level", "--outdir", "outputs/seasonality/s1_level"]),
        
        # S1 Delta
        ("s1_overlay.py", ["--year", "2024", "--metric", "diff", "--outdir", "outputs/seasonality/s1_delta"]),
        ("s1_average_by_bdom.py", ["--year", "2024", "--metric", "diff", "--outdir", "outputs/seasonality/s1_delta"]),
        
        # S2 Level
        ("s2_overlay.py", ["--year", "2024", "--metric", "level", "--outdir", "outputs/seasonality/s2_level"]),
        ("s2_average_by_bdom.py", ["--year", "2024", "--metric", "level", "--outdir", "outputs/seasonality/s2_level"]),
        
        # S2 Delta
        ("s2_overlay.py", ["--year", "2024", "--metric", "diff", "--outdir", "outputs/seasonality/s2_delta"]),
        ("s2_average_by_bdom.py", ["--year", "2024", "--metric", "diff", "--outdir", "outputs/seasonality/s2_delta"]),

        # Regressions
        ("s1_eom_trend.py", []),
        ("s1_eom_intra_month_slope.py", []),
    ]
    
    success_count = 0
    for script_name, args in scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"WARNING: {script_path} does not exist")
            continue
            
        if run_script(script_path, args):
            success_count += 1
            
    print(f"\nCompleted {success_count}/{len(scripts)} scripts.")
    if success_count == len(scripts):
        print("All S1/S2 scripts ran successfully.")
        return 0
    else:
        print("Some scripts failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
