"""Opt-in licensed simulator check, never run by normal test discovery.

Example: python tests\\validate_eclipse_design.py --runner C:\\APPL\\ECL\\macros\\eclrun.exe
         --output C:\\scratch\\eclipse-design-check --case oil
"""
import argparse
from pathlib import Path
import os
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import eclipse_design as D, models as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=["oil", "gas", "mfhw"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    p = M.WellTestParams(C=0, xf=20, n_frac=2, f_space=60,
                        model="mfhw" if args.case == "mfhw" else "homogeneous")
    setup = D.DeckSetup(nx=9, ny=9, nz=1, half_extent_ft=500, report_points_per_period=8)
    gas = {"T_F": 220.0, "gamma_g": 0.68, "cf": 4e-6} if args.case == "gas" else None
    fluid = "Gas" if gas else "Oil"
    stem = "WTW_" + args.case.upper()
    deck, _ = D.build_deck(p, setup, "PROD", 10, 1, 0.5, fluid, 14.7, gas)
    (args.output / (stem + ".DATA")).write_text(deck, encoding="ascii")
    report = args.output / (stem + ".PRT")
    if report.exists():
        report.unlink()
    env = os.environ.copy()
    env["ECLPATH"] = str(args.runner.parent.parent)
    command = [str(args.runner), "--maxruntime", "120", "--summary-conversion", "no", "eclipse", stem]
    with (args.output / (stem + ".runner.log")).open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=args.output, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=180, check=False)
    print(f"RUNNER_EXIT={result.returncode}", flush=True)
    print(f"CASE_DIRECTORY={args.output}", flush=True)
    if not report.exists():
        print("SIMULATOR_CHECK=FAIL: no PRT report", flush=True)
        return result.returncode or 1
    text = report.read_text(encoding="utf-8", errors="replace")
    counts = re.findall(r"^\s*Errors\s+(\d+)", text, re.MULTILINE)
    if not counts or int(counts[-1]) != 0:
        print(f"SIMULATOR_CHECK=FAIL: Errors={counts[-1] if counts else 'missing summary'}", flush=True)
        return result.returncode or 1
    print("SIMULATOR_CHECK=PASS: Errors=0", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
