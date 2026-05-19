from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solarchain_eval.config import load_config
from solarchain_eval.figures import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SolarChain-Eval figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    written = make_figures(args.run_dir, args.figures_dir or config.figures_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

