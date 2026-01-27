import argparse
import json
import re
from pathlib import Path


METRICS = [
    "accuracy",
    "completion",
    "nc1",
    "nc2",
    "acc_med",
    "comp_med",
    "nc1_med",
    "nc2_med",
    "abs_rel",
    "sq_rel",
    "rmse",
    "delta_1_25",
    "delta_1_25_2",
]


def parse_metrics(text: str) -> dict:
    out = {}
    for key in METRICS:
        m = re.findall(rf"^{re.escape(key)}\\s+([-+0-9.eE]+)\\s*$", text, flags=re.MULTILINE)
        if not m:
            continue
        try:
            out[key] = float(m[-1])
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=str)
    ap.add_argument("--json", action="store_true", help="print JSON")
    args = ap.parse_args()

    text = Path(args.log_path).read_text(errors="ignore")
    metrics = parse_metrics(text)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    # pretty print
    for k in METRICS:
        if k in metrics:
            print(f"{k}: {metrics[k]:.6f}")


if __name__ == "__main__":
    main()

