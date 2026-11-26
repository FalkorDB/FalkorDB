"""Store one perf run's files as the dashboard reads them: `bench nightly store`.

perf-dashboard.yml (a night, into the gh-pages archive) and perf-comment.yml (a `/perf` run, into a
site of its own) both call it:

  bench nightly store [--nightly <nightly.json>] --run-id <id> --data <dir> --trigger T --scope S

It downloads each check's newest `perf-result-<check>` artifact of run <id> (with `gh`, under a
timeout, tried again when a download stalls; without --nightly, the run's nightly.json from its
`perf-nightly` artifact too) and writes
<dir>/<date>-<id>/: nightly.json as given, run.json {"trigger": T, "scope": S}, and per check
<check>/ holding what the run's page reads or links: result.json, raw/rows.csv and the runner's
record (perf-vm.json or perf-hosted.json), as .json/.csv/.txt/.log only. That is about 0.5 MB of
a full night's 80 MB: the rest stays in the run's artifacts. With --full (reading a run
locally), every file its result.json lists instead, of any type (relative paths, none through
".."). A folder of the same run already there (a re-run of the night, under any date) is replaced, but never by a store that
lost a check the old folder has: then nothing changes and it exits 1. Prints the folder's name.

Standard library only (with nightly.dashboard, which is too).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from pathlib import PurePosixPath

from falkorbench.nightly import dashboard

RECORDS = ("result.json", "perf-vm.json", "perf-hosted.json")
# What a run's page reads or links, per check, and the only types it serves: a /perf candidate is
# pull request code, so nothing it can influence is served as a page or a script.
PAGE_FILES = ("result.json", "raw/rows.csv", "raw/summary.csv", "perf-vm.json", "perf-hosted.json")
SHOWN = (".json", ".csv", ".txt", ".log")
# A page file larger than this is left out (with a warning): one runaway file would count
# against Pages' 1 GB in every deploy.
MAX_PAGE_FILE = 5 * 1024 * 1024
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
RUN_ID_RE = re.compile(r"\d{1,20}")
# Seconds for one `gh` call (a listing; an artifact's zip), and before each retry. Tests set them.
LIST_TIMEOUT = 60.0
ZIP_TIMEOUT = 600.0
RETRY_DELAYS = (5.0, 15.0)

Runner = Callable[..., subprocess.CompletedProcess]


class StoreError(Exception):
    pass


def warn(message: str) -> None:
    print(f"::warning::perf dashboard: {message}", file=sys.stderr)


def relative(path: object) -> bool:
    """A path the store may copy: relative, not empty, never through ".."."""
    if not isinstance(path, str) or not path or "\0" in path or path.startswith("/"):
        return False
    return ".." not in PurePosixPath(path).parts


def extract(archive: Path, dest: Path) -> None:
    """Unzip into `dest`. A member that would land outside it (absolute, or through "..")
    fails the whole archive, as `unzip` did: no artifact upload writes one."""
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        unsafe = [m.filename for m in members if not relative(m.filename)]
        if unsafe:
            raise zipfile.BadZipFile(f"member {unsafe[0]!r} would land outside")
        for member in members:
            zipped.extract(member, dest)


def _gh(
    argv: list[str], run: Runner, timeout: float, **kwargs
) -> subprocess.CompletedProcess | None:
    """One `gh` call under a timeout, tried again (after RETRY_DELAYS) when it fails or stalls:
    GitHub's artifact downloads stall now and then. GitHub's 404 (no such run or artifact) is
    returned as it is, not tried again; any other error (a 403 rate limit, a 5xx) is tried again.
    None when every attempt failed."""
    for delay in (*RETRY_DELAYS, None):
        try:
            done = run(argv, timeout=timeout, check=False, **kwargs)
            said = (
                done.stderr.decode(errors="replace")
                if isinstance(done.stderr, bytes)
                else str(done.stderr or "")
            )
            if done.returncode == 0 or "HTTP 404" in said:
                return done
            why = f"exit {done.returncode}"
        except subprocess.TimeoutExpired:
            why = f"no answer in {timeout:.0f} s"
        if delay is None:
            warn(f"gh {argv[2] if len(argv) > 2 else ''}: {why}; giving up")
            return None
        warn(f"gh {argv[2] if len(argv) > 2 else ''}: {why}; trying again")
        time.sleep(delay)
        handle = kwargs.get("stdout")
        if hasattr(handle, "seek"):
            handle.seek(0)
            handle.truncate()
    return None


def download(repo: str, run_id: str, name: str, dest: Path, run: Runner = subprocess.run) -> str:
    """Run <run_id>'s newest unexpired artifact <name>, extracted into <dest>: "ok", "missing"
    (the run has none) or "failed" (it has one that did not download). After a re-run several
    artifacts share a name: the newest by id is the last attempt's."""
    listing = _gh(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/artifacts?name={name}&per_page=100"],
        run,
        LIST_TIMEOUT,
        capture_output=True,
        text=True,
    )
    if listing is None:
        warn(f"{name}: could not list its artifacts")
        return "failed"
    if listing.returncode != 0:  # GitHub's 404: the run is not there
        warn(f"{name}: GitHub has no such run ({str(listing.stderr).strip()[:200]})")
        return "missing"
    try:
        body = json.loads(listing.stdout)
    except ValueError:
        body = None
    if not isinstance(body, dict) or not isinstance(body.get("artifacts"), list):
        # An answer that is not a listing says nothing about the run: never "it has none".
        warn(f"{name}: GitHub's listing of its artifacts is not one")
        return "failed"
    artifacts = body["artifacts"]
    ids = [
        a["id"]
        for a in artifacts
        if isinstance(a, dict) and isinstance(a.get("id"), int) and not a.get("expired")
    ]
    if not ids:
        warn(f"{name}: no artifact")
        return "missing"
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / f"{name}.zip"
        with archive.open("wb") as handle:
            done = _gh(
                ["gh", "api", f"repos/{repo}/actions/artifacts/{max(ids)}/zip"],
                run,
                ZIP_TIMEOUT,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        try:
            if done is None:
                raise zipfile.BadZipFile("gh did not deliver it")
            dest.mkdir(parents=True, exist_ok=True)
            extract(archive, dest)
        except (OSError, EOFError, zipfile.BadZipFile, zlib.error) as err:
            # A partly extracted artifact must not pass for a result.
            shutil.rmtree(dest, ignore_errors=True)
            warn(f"{name}: artifact {max(ids)} did not download ({err})")
            return "failed"
    return "ok"


def fetch(repo: str, run_id: str, name: str, dest: Path, run: Runner = subprocess.run) -> str:
    """download()'s answer: "ok", "missing" or "failed"."""
    return download(repo, run_id, name, dest, run)


def check_files(src: Path) -> list[str]:
    """The files of one check to store: its records, then what its result.json lists."""
    files = [name for name in RECORDS if (src / name).is_file()]
    try:
        result = json.loads((src / "result.json").read_text(encoding="utf-8"))
        listed = result.get("artifacts") if isinstance(result, dict) else None
    except (OSError, UnicodeError, ValueError):
        listed = None
    files += [a for a in listed if isinstance(a, str)] if isinstance(listed, list) else []
    return sorted(dict.fromkeys(files))


def copy_page_files(src: Path, dest: Path) -> None:
    """Only what the run page reads or links (PAGE_FILES: also ldbc's raw/summary.csv, which
    `bench nightly failures` reads), of the types in SHOWN, each at most MAX_PAGE_FILE."""
    for name in PAGE_FILES:
        source = src / name
        if not (source.is_file() and not source.is_symlink() and source.suffix in SHOWN):
            continue
        if source.stat().st_size > MAX_PAGE_FILE:
            warn(f"{src.name}/{name}: {source.stat().st_size} bytes, over the page limit: left out")
            continue
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest / name)


def trim(src: Path, dest: Path) -> int:
    """Copy every run under <src> (a data folder, as stored) to <dest> with only its pages'
    files: what a deploy publishes, whatever the archive holds. Returns the number of runs."""
    count = 0
    for run in sorted(p for p in src.iterdir() if p.is_dir()) if src.is_dir() else []:
        if not dashboard.FOLDER_RE.fullmatch(run.name):
            continue
        (dest / run.name).mkdir(parents=True, exist_ok=True)
        for name in ("nightly.json", "run.json"):
            if (run / name).is_file() and not (run / name).is_symlink():
                shutil.copyfile(run / name, dest / run.name / name)
        for check in sorted(p for p in run.iterdir() if p.is_dir()):
            if dashboard.CHECK_RE.fullmatch(check.name):
                copy_page_files(check, dest / run.name / check.name)
        count += 1
    return count


def copy_check(src: Path, dest: Path) -> None:
    root = src.resolve()
    for name in check_files(src):
        source = src / name
        if not relative(name) or not source.resolve().is_relative_to(root):
            continue
        target = dest / name
        try:
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            else:
                raise FileNotFoundError(name)
        except OSError:
            warn(f"{dest.name}/{name} did not copy")


def store(
    night_path: Path,
    run_id: str,
    data: Path,
    trigger: str,
    scope: str,
    download: Callable[[str, Path], str | bool],
    full: bool = False,
) -> str:
    """Lay out run <run_id> under <data>; returns its folder's name. `download(name, dest)`
    extracts one artifact of the run into dest and says "ok"; "missing" (or False) when the run
    has none (the check left no result), which the store tolerates; "failed" when it has one that
    did not download, which stops the store with nothing changed, so a re-run tries again. Per check, only what the run page
    needs (copy_page_files), unless `full`: every file its result.json lists."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise StoreError(f"run id {run_id!r} is not a number")
    try:
        night = json.loads(night_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as err:
        raise StoreError(f"cannot read {night_path}: {err}") from err
    date = night.get("date") if isinstance(night, dict) else None
    if not (isinstance(date, str) and DATE_RE.fullmatch(date)):
        raise StoreError(f"{night_path}: date {date!r} is not YYYY-MM-DD")
    folder = f"{date}-{run_id}"
    if not dashboard.FOLDER_RE.fullmatch(folder):
        raise StoreError(f"folder {folder!r} is not <date>-<run id>")
    checks = [
        c.get("check")
        for c in (night.get("checks") if isinstance(night.get("checks"), list) else [])
        if isinstance(c, dict) and isinstance(c.get("check"), str)
    ]
    checks = [name for name in dict.fromkeys(checks) if dashboard.CHECK_RE.fullmatch(name)]

    # A re-run of the night replaces what its first attempt stored (under any date).
    data.mkdir(parents=True, exist_ok=True)
    replaced = sorted(p for p in data.glob(f"*-{run_id}") if p.is_dir())
    stored = {child.name for old in replaced for child in old.iterdir()}
    # A re-run that lists fewer checks (its registry did not load, say) must not delete the
    # stored folders of the checks it dropped.
    dropped = sorted(
        child.name
        for old in replaced
        for child in old.iterdir()
        if child.is_dir() and dashboard.CHECK_RE.fullmatch(child.name) and child.name not in checks
    )
    if dropped:
        names = ", ".join(p.name for p in replaced)
        raise StoreError(f"not replacing {names}: this run does not list {', '.join(dropped)}")

    with tempfile.TemporaryDirectory() as scratch:
        results = Path(scratch)
        fetched = {}
        for check in checks:
            dest = results / f"perf-result-{check}"
            got = download(f"perf-result-{check}", dest)
            if (got is True or got == "ok") and (dest / "result.json").is_file():
                fetched[check] = dest
                continue
            if got == "failed":
                raise StoreError(f"perf-result-{check} did not download: run it again")
            if got is True or got == "ok":
                # An artifact with no result.json (the VM died before the check wrote it): its
                # runner's record is still stored, but it never replaces a stored result.
                fetched[check] = dest
                if check in stored:
                    names = ", ".join(p.name for p in replaced)
                    raise StoreError(
                        f"not replacing {names}: perf-result-{check} has no result.json"
                    )
                continue
            warn(f"no perf-result-{check}")
            # A missing download must not delete what a stored run has; a check the stored run
            # never had (it left no artifact then either) is no loss.
            if check in stored:
                names = ", ".join(p.name for p in replaced)
                raise StoreError(f"not replacing {names} without perf-result-{check}")
        for old in replaced:
            shutil.rmtree(old)
        dest = data / folder
        dest.mkdir(parents=True)
        shutil.copyfile(night_path, dest / "nightly.json")
        meta = {"trigger": trigger, "scope": scope}
        (dest / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for check, src in fetched.items():
            (dest / check).mkdir()
            if full:
                copy_check(src, dest / check)
            else:
                copy_page_files(src, dest / check)
    return folder


def main(argv: list[str] | None = None, run: Runner = subprocess.run) -> int:
    parser = argparse.ArgumentParser(prog="bench nightly store", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--nightly", type=Path, help="the run's nightly.json; default: its perf-nightly artifact"
    )
    parser.add_argument("--run-id", required=True, help="the run whose perf-result-* to download")
    parser.add_argument("--data", required=True, type=Path, help="holds one <date>-<id>/ per run")
    parser.add_argument("--trigger", required=True, help="run.json trigger, e.g. nightly")
    parser.add_argument("--scope", required=True, help="run.json scope, e.g. main or pr-<N>")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "FalkorDB/FalkorDB"))
    parser.add_argument(
        "--full",
        action="store_true",
        help="every file each check lists, of any type (reading a run locally); default: the page's",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        night = args.nightly
        if night is None:
            got = download(args.repo, args.run_id, "perf-nightly", Path(scratch) / "night", run)
            if got != "ok":
                why = (
                    "the run has no perf-nightly artifact: it never merged its checks (cancelled, "
                    "or its resolve job failed), or the artifact expired"
                    if got == "missing"
                    else "its perf-nightly artifact did not download (see the warnings above): try again"
                )
                print(f"::error::perf dashboard: run {args.run_id}: {why}", file=sys.stderr)
                return 1
            night = Path(scratch) / "night" / "nightly.json"
        try:
            folder = store(
                night,
                args.run_id,
                args.data,
                args.trigger,
                args.scope,
                lambda name, dest: fetch(args.repo, args.run_id, name, dest, run),
                full=args.full,
            )
        except StoreError as err:
            print(f"::error::perf dashboard: {err}", file=sys.stderr)
            return 1
    print(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
