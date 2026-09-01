#!/usr/bin/env python3
"""Walk back through the git history, build each commit and run the UCI `bench`.

Every commit is checked out into a throwaway git worktree, built with a
CMake preset and benchmarked a few times. The last line the engine prints
for a bench looks like

    2228454 nodes 1148095 nps

so that is what gets scraped. The NPS of the runs is averaged, the node
count is reported as is (it should not move between runs of the same
commit, and it is flagged when it does).

Usage:
    python bench_history.py                     # last 10 commits, 3 runs each
    python bench_history.py -n 20 --runs 5
    python bench_history.py --start v2.0 -n 5 --preset linux-clang
    python bench_history.py -n 10 --csv bench.csv
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# "2228454 nodes 1148095 nps" - the summary bench prints once it is done.
BENCH_RE = re.compile(r"(\d+)\s+nodes\s+(\d+)\s+nps")

# Fields of one `git log` record, split on characters that cannot show up in
# a commit subject.
LOG_FORMAT = "%H%x1f%h%x1f%ad%x1f%s"


class BenchError(Exception):
    """A commit could not be built or benchmarked. Reported, never fatal."""


def git(args, cwd, capture=True, check=True):
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        raise BenchError(f"git {' '.join(args)} failed: {detail}")
    return (result.stdout or "").strip() if capture else ""


def repo_root(start):
    return Path(git(["rev-parse", "--show-toplevel"], cwd=start))


def collect_commits(repo, start, count):
    """The `count` commits reachable from `start`, newest first."""
    out = git(
        ["log", f"-n{count}", f"--format={LOG_FORMAT}", "--date=short", start],
        cwd=repo,
    )
    commits = []
    for line in out.splitlines():
        sha, short, date, subject = line.split("\x1f", 3)
        commits.append({"sha": sha, "short": short, "date": date, "subject": subject})
    return commits


def ensure_worktree(repo, path, ref):
    """Create (or reuse) a detached worktree so the real checkout is untouched."""
    git(["worktree", "prune"], cwd=repo)
    if (path / ".git").exists():
        print(f"Reusing worktree {path}")
        return
    if path.exists() and any(path.iterdir()):
        raise BenchError(f"{path} exists and is not an empty directory or a worktree")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating worktree {path}")
    git(["worktree", "add", "--detach", str(path), ref], cwd=repo)


def remove_worktree(repo, path):
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=str(repo),
        text=True,
        capture_output=True,
        check=False,
    )
    shutil.rmtree(path, ignore_errors=True)


def run_step(cmd, cwd, verbose, what):
    """Run a build step, keeping its output unless it fails or -v was given."""
    print(f"    {what}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=None if verbose else subprocess.PIPE,
        stderr=subprocess.STDOUT if not verbose else None,
        check=False,
    )
    if result.returncode != 0:
        if not verbose and result.stdout:
            sys.stderr.write(result.stdout[-4000:] + "\n")
        raise BenchError(f"{what} failed with exit code {result.returncode}")


def build(worktree, args):
    """Configure and build the Zaphod target, returning the binary path."""
    build_dir = worktree / "build" / args.preset
    if args.clean_build:
        shutil.rmtree(build_dir, ignore_errors=True)

    extra = []
    if args.network_file:
        extra.append(f"-DZAPHOD_NETWORK_FILE={Path(args.network_file).resolve()}")
    extra += args.cmake_arg

    if (worktree / "CMakePresets.json").exists():
        configure = ["cmake", "--preset", args.preset] + extra
        compile_cmd = ["cmake", "--build", "--preset", args.preset, "--target", args.target]
    else:
        # Old enough to predate CMakePresets.json - configure by hand into the
        # same directory the preset would have used.
        configure = [
            "cmake", "-S", ".", "-B", str(build_dir), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        ] + extra
        compile_cmd = ["cmake", "--build", str(build_dir), "--target", args.target]

    if args.jobs:
        compile_cmd += ["-j", str(args.jobs)]

    run_step(configure, worktree, args.verbose, "configure")
    run_step(compile_cmd, worktree, args.verbose, "build")

    for candidate in (build_dir / args.target, build_dir / f"{args.target}.exe"):
        if candidate.exists():
            return candidate
    raise BenchError(f"built, but no {args.target} binary under {build_dir}")


def run_bench(binary, worktree, args):
    """One bench run. Returns (nodes, nps, seconds)."""
    started = time.monotonic()
    try:
        result = subprocess.run(
            [str(binary)] + args.engine_arg,
            cwd=str(worktree),
            input="bench\nquit\n",
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise BenchError(f"bench did not finish within {args.timeout}s")
    elapsed = time.monotonic() - started

    if result.returncode != 0:
        raise BenchError(f"engine exited with code {result.returncode}")

    matches = BENCH_RE.findall(result.stdout)
    if not matches:
        tail = "\n".join(result.stdout.strip().splitlines()[-5:])
        raise BenchError(f"no '<nodes> nodes <nps> nps' line in the output. Tail:\n{tail}")

    # The per position lines print nodes and NPS in the other order, so the
    # summary is simply the last thing that matches.
    nodes, nps = matches[-1]
    return int(nodes), int(nps), elapsed


def bench_commit(repo, worktree, commit, args):
    """Check out, build and benchmark one commit. Never raises."""
    entry = dict(commit)
    entry.update(runs=[], nodes=None, nps_avg=None, error=None)

    print(f"\n=== {commit['short']}  {commit['date']}  {commit['subject']}")
    try:
        git(["checkout", "--detach", "--force", commit["sha"]], cwd=worktree)
        binary = build(worktree, args)

        node_counts = []
        for run in range(1, args.runs + 1):
            nodes, nps, elapsed = run_bench(binary, worktree, args)
            node_counts.append(nodes)
            entry["runs"].append(nps)
            print(f"    run {run}/{args.runs}: {nodes} nodes {nps} nps ({elapsed:.1f}s)")

        entry["nodes"] = node_counts[0]
        entry["nps_avg"] = statistics.mean(entry["runs"])
        if len(set(node_counts)) > 1:
            entry["node_counts"] = node_counts
            print(f"    note: node count varied between runs: {node_counts}")
    except BenchError as exc:
        entry["error"] = str(exc)
        print(f"    FAILED: {exc}")
    return entry


def format_table(results, oldest_first):
    """The summary table, with each row compared against the older commit."""
    ordered = list(reversed(results)) if oldest_first else list(results)

    subject_width = max([len(r["subject"]) for r in ordered] + [7])
    subject_width = min(subject_width, 60)
    header = (
        f"{'commit':<9}{'date':<12}{'nodes':>12}{'avg nps':>12}"
        f"{'min nps':>12}{'max nps':>12}{'spread':>8}{'vs older':>10}  subject"
    )
    lines = [header, "-" * len(header)]

    for row in ordered:
        subject = row["subject"]
        if len(subject) > subject_width:
            subject = subject[: subject_width - 1] + "…"

        if row["error"]:
            lines.append(
                f"{row['short']:<9}{row['date']:<12}{'FAILED':>12}{'-':>12}"
                f"{'-':>12}{'-':>12}{'-':>8}{'-':>10}  {subject}"
            )
            continue

        runs = row["runs"]
        spread = (max(runs) - min(runs)) / row["nps_avg"] * 100 if row["nps_avg"] else 0.0
        delta = f"{row['delta_pct']:+.1f}%" if row.get("delta_pct") is not None else "-"
        lines.append(
            f"{row['short']:<9}{row['date']:<12}{row['nodes']:>12}"
            f"{round(row['nps_avg']):>12}{min(runs):>12}{max(runs):>12}"
            f"{spread:>7.1f}%{delta:>10}  {subject}"
        )
    return "\n".join(lines)


def annotate_deltas(results):
    """Percentage change in average NPS against the next older tested commit.

    `results` is newest first, so the older neighbour is the next entry.
    """
    for index, row in enumerate(results):
        row["delta_pct"] = None
        if row["error"] or row["nps_avg"] is None:
            continue
        for older in results[index + 1:]:
            if older["error"] or not older["nps_avg"]:
                continue
            row["delta_pct"] = (row["nps_avg"] - older["nps_avg"]) / older["nps_avg"] * 100
            break


def write_csv(path, results):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha", "short", "date", "subject", "nodes", "avg_nps",
                         "min_nps", "max_nps", "runs", "delta_pct", "error"])
        for row in results:
            runs = row["runs"]
            writer.writerow([
                row["sha"], row["short"], row["date"], row["subject"],
                row["nodes"] if row["nodes"] is not None else "",
                round(row["nps_avg"]) if row["nps_avg"] is not None else "",
                min(runs) if runs else "",
                max(runs) if runs else "",
                " ".join(str(nps) for nps in runs),
                f"{row['delta_pct']:.2f}" if row.get("delta_pct") is not None else "",
                row["error"] or "",
            ])
    print(f"Wrote {path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build and bench the last N commits, averaging the NPS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--commits", type=int, default=10,
                        help="how many commits back from --start to test")
    parser.add_argument("--start", default="HEAD",
                        help="commit to start walking back from")
    parser.add_argument("--runs", type=int, default=3,
                        help="bench runs per commit, averaged")
    parser.add_argument("--preset", default="linux-gcc",
                        help="CMake configure/build preset")
    parser.add_argument("--target", default="Zaphod", help="CMake target to build")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                        help="parallel build jobs")
    parser.add_argument("--worktree",
                        help="worktree directory to build in. Reused and kept when "
                             "given, which saves rebuilding from scratch and "
                             "re-downloading the net (default: a temporary one)")
    parser.add_argument("--keep-worktree", action="store_true",
                        help="do not remove the temporary worktree afterwards")
    parser.add_argument("--clean-build", action="store_true",
                        help="wipe the build directory before every commit")
    parser.add_argument("--network-file",
                        help="NNUE network to embed, passed as ZAPHOD_NETWORK_FILE. "
                             "Skips the download during configure")
    parser.add_argument("--cmake-arg", action="append", default=[], metavar="ARG",
                        help="extra argument for the configure step (repeatable)")
    parser.add_argument("--engine-arg", action="append", default=[], metavar="ARG",
                        help="extra argument for the engine binary (repeatable)")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="seconds a single bench run may take")
    parser.add_argument("--csv", help="also write the results to this CSV file")
    parser.add_argument("--json", help="also write the results to this JSON file")
    parser.add_argument("--oldest-first", action="store_true",
                        help="print the table oldest commit first")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the commits that would be tested and stop")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the full configure and build output")
    args = parser.parse_args(argv)

    if args.commits < 1:
        parser.error("--commits must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)

    try:
        repo = repo_root(Path(__file__).resolve().parent)
        commits = collect_commits(repo, args.start, args.commits)
    except BenchError as exc:
        sys.exit(str(exc))

    if not commits:
        sys.exit(f"No commits found for {args.start}")
    if len(commits) < args.commits:
        print(f"Only {len(commits)} commits reachable from {args.start}")

    print(f"Benchmarking {len(commits)} commits, {args.runs} runs each:")
    for commit in commits:
        print(f"  {commit['short']}  {commit['date']}  {commit['subject']}")
    if args.dry_run:
        return 0

    if args.worktree:
        worktree = Path(args.worktree).resolve()
        keep = True
    else:
        worktree = repo.parent / f".zaphod-bench-history-{os.getpid()}"
        keep = args.keep_worktree

    results = []
    try:
        ensure_worktree(repo, worktree, commits[0]["sha"])
        for commit in commits:
            results.append(bench_commit(repo, worktree, commit, args))
    except BenchError as exc:
        print(f"\nAborted: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted, reporting what finished so far.", file=sys.stderr)
    finally:
        if not keep:
            remove_worktree(repo, worktree)
        elif worktree.exists():
            print(f"\nWorktree left at {worktree}")

    if not results:
        return 1

    annotate_deltas(results)
    print("\n" + format_table(results, args.oldest_first))

    benched = [row for row in results if row["nps_avg"] is not None]
    if len(benched) > 1:
        fastest = max(benched, key=lambda row: row["nps_avg"])
        slowest = min(benched, key=lambda row: row["nps_avg"])
        print(f"\nFastest: {fastest['short']} at {round(fastest['nps_avg'])} nps"
              f"  ({fastest['subject']})")
        print(f"Slowest: {slowest['short']} at {round(slowest['nps_avg'])} nps"
              f"  ({slowest['subject']})")
    failed = [row for row in results if row["error"]]
    if failed:
        print(f"\n{len(failed)} commit(s) failed: "
              + ", ".join(row["short"] for row in failed))

    if args.csv:
        write_csv(args.csv, results)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote {args.json}")

    return 0 if benched else 1


if __name__ == "__main__":
    sys.exit(main())

