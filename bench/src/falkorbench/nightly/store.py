"""Store one perf run's files as the dashboard reads them: `bench nightly store`.

perf-dashboard.yml (a night, into the gh-pages archive) and perf-comment.yml (a `/perf` run, into a
site of its own) both call it:

  bench nightly store --nightly <nightly.json> --run-id <id> --data <dir> --trigger T --scope S

It downloads each check's newest `perf-result-<check>` artifact of run <id> (with `gh`) and writes
<dir>/<date>-<id>/: nightly.json as given, run.json {"trigger": T, "scope": S}, and per check
<check>/ holding its result.json, the runner's record (perf-vm.json or perf-hosted.json) and the
files its result.json lists as artifacts (relative paths, none through ".."). A folder of the same
run already there (a re-run of the night, under any date) is replaced, but never by a store that
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
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from pathlib import PurePosixPath

from falkorbench.nightly import dashboard

RECORDS = ("result.json", "perf-vm.json", "perf-hosted.json")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
RUN_ID_RE = re.compile(r"\d{1,20}")

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


def fetch(repo: str, run_id: str, name: str, dest: Path, run: Runner = subprocess.run) -> bool:
    """Run <run_id>'s newest unexpired artifact <name>, extracted into <dest>. After a re-run
    several artifacts share a name: the newest by id is the last attempt's."""
    listing = run(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/artifacts?name={name}&per_page=100"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        warn(f"{name}: could not list its artifacts")
        return False
    try:
        artifacts = json.loads(listing.stdout).get("artifacts") or []
    except (ValueError, AttributeError):
        artifacts = []
    ids = [
        a["id"]
        for a in artifacts
        if isinstance(a, dict) and isinstance(a.get("id"), int) and not a.get("expired")
    ]
    if not ids:
        warn(f"{name}: no artifact")
        return False
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / f"{name}.zip"
        with archive.open("wb") as handle:
            done = run(
                ["gh", "api", f"repos/{repo}/actions/artifacts/{max(ids)}/zip"],
                stdout=handle,
                stderr=subprocess.PIPE,
                check=False,
            )
        try:
            if done.returncode != 0:
                raise zipfile.BadZipFile(f"gh exited {done.returncode}")
            dest.mkdir(parents=True, exist_ok=True)
            extract(archive, dest)
        except (OSError, EOFError, zipfile.BadZipFile, zlib.error) as err:
            # A partly extracted artifact must not pass for a result.
            shutil.rmtree(dest, ignore_errors=True)
            warn(f"{name}: artifact {max(ids)} did not download ({err})")
            return False
    return True


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
    download: Callable[[str, Path], bool],
) -> str:
    """Lay out run <run_id> under <data>; returns its folder's name. `download(name, dest)`
    extracts one artifact of the run into dest, or says False."""
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

    with tempfile.TemporaryDirectory() as scratch:
        results = Path(scratch)
        fetched = {}
        for check in checks:
            dest = results / f"perf-result-{check}"
            if download(f"perf-result-{check}", dest):
                fetched[check] = dest
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
            copy_check(src, dest / check)
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench nightly store", description=__doc__.split("\n")[0])
    parser.add_argument("--nightly", required=True, type=Path, help="the run's nightly.json")
    parser.add_argument("--run-id", required=True, help="the run whose perf-result-* to download")
    parser.add_argument("--data", required=True, type=Path, help="holds one <date>-<id>/ per run")
    parser.add_argument("--trigger", required=True, help="run.json trigger, e.g. nightly")
    parser.add_argument("--scope", required=True, help="run.json scope, e.g. main or pr-<N>")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "FalkorDB/FalkorDB"))
    args = parser.parse_args(argv)
    try:
        folder = store(
            args.nightly,
            args.run_id,
            args.data,
            args.trigger,
            args.scope,
            lambda name, dest: fetch(args.repo, args.run_id, name, dest),
        )
    except StoreError as err:
        print(f"::error::perf dashboard: {err}", file=sys.stderr)
        return 1
    print(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
