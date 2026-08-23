#!/usr/bin/env python3
"""Generate a JSON tuning config from ZAP_TUNABLE_INT declarations in params.h.

Usage:
    python gen_tune_config.py params.h -o tune.json
    python gen_tune_config.py params.h --values best.txt -o tune.json
"""

import argparse
import json
import re
import sys
from collections import OrderedDict

# ZAP_TUNABLE_INT(Name, Value, Min, Max, Step)
TUNABLE_RE = re.compile(
    r"""ZAP_TUNABLE_INT\s*\(\s*
        (?P<name>[A-Za-z_]\w*)\s*,\s*
        (?P<value>[+-]?\d+)\s*,\s*
        (?P<min>[+-]?\d+)\s*,\s*
        (?P<max>[+-]?\d+)\s*,\s*
        (?P<step>[+-]?\d+)\s*
        \)""",
    re.VERBOSE,
)

# Line/block comments, so commented-out params are ignored.
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def strip_comments(text: str) -> str:
    # Replace with newlines to keep line numbering intact for error messages.
    return COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def parse_header(text: str):
    params = OrderedDict()
    for lineno, line in enumerate(strip_comments(text).splitlines(), 1):
        # Skip the macro definition itself.
        if line.lstrip().startswith("#define"):
            continue
        m = TUNABLE_RE.search(line)
        if not m:
            continue
        name = m.group("name")
        value, lo, hi, step = (int(m.group(k)) for k in ("value", "min", "max", "step"))
        if name in params:
            print(f"warning: duplicate parameter '{name}' on line {lineno}", file=sys.stderr)
        if lo > hi:
            print(f"warning: {name}: min {lo} > max {hi}", file=sys.stderr)
        if not lo <= value <= hi:
            print(f"warning: {name}: value {value} outside [{lo}, {hi}]", file=sys.stderr)
        if step <= 0:
            print(f"warning: {name}: non-positive step {step}", file=sys.stderr)
        params[name] = OrderedDict(
            [("value", value), ("min_value", lo), ("max_value", hi), ("step", step)]
        )
    return params


def load_overrides(path: str):
    """Read tuned values from JSON ({name: value} or the config format) or
    plain 'name value' / 'name=value' / 'name: value' lines."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        return {
            k: int(round(float(v["value"] if isinstance(v, dict) else v)))
            for k, v in data.items()
        }

    overrides = {}
    for line in raw.splitlines():
        line = line.split("//")[0].split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*[:=, \t]\s*([+-]?\d+(?:\.\d+)?)$", line)
        if m:
            overrides[m.group(1)] = int(round(float(m.group(2))))
    return overrides


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("header", nargs="?", default="params.h", help="path to params.h")
    ap.add_argument("-o", "--output", help="output JSON file (default: stdout)")
    ap.add_argument("--values", help="file with tuned values overriding the header defaults")
    ap.add_argument("--clamp", action="store_true",
                    help="clamp overridden values into [min, max]")
    ap.add_argument("--sort", action="store_true", help="sort parameters by name")
    args = ap.parse_args()

    with open(args.header, encoding="utf-8") as f:
        params = parse_header(f.read())

    if not params:
        sys.exit(f"error: no ZAP_TUNABLE_INT declarations found in {args.header}")

    if args.values:
        overrides = load_overrides(args.values)
        for name, value in overrides.items():
            if name not in params:
                print(f"warning: override for unknown parameter '{name}'", file=sys.stderr)
                continue
            p = params[name]
            if args.clamp:
                value = max(p["min_value"], min(p["max_value"], value))
            elif not p["min_value"] <= value <= p["max_value"]:
                print(f"warning: {name}: override {value} outside "
                      f"[{p['min_value']}, {p['max_value']}]", file=sys.stderr)
            p["value"] = value

    if args.sort:
        params = OrderedDict(sorted(params.items()))

    text = json.dumps(params, indent=4) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(params)} parameters to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()