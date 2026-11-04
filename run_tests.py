"""Run every regression suite, including the stand-in UI smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", help="Suite names without .py; default: all suites")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    names = args.suites or [p.stem for p in sorted((root / "tests").glob("test_*.py"))] + ["ui_smoke"]
    failures = []
    for name in names:
        path = root / "tests" / f"{name}.py"
        if path.parent != root / "tests" or not path.is_file():
            parser.error(f"Unknown suite: {name}")
        print(f"== {name}", flush=True)
        result = subprocess.run([sys.executable, "-I", str(path)], cwd=root, check=False)
        if result.returncode:
            failures.append(name)
    print(f"\nSuites: {len(names) - len(failures)} passed, {len(failures)} failed", flush=True)
    if failures:
        print("Failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
