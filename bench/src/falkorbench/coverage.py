"""How much of the graph crate the query set actually reaches.

Builds an instrumented debug module, runs every query once, and reports line
coverage of `graph/src` (excluding the generated GraphBLAS FFI).

This is a **validator of the query set**, not a coverage gate: it reports a
percentage and enforces no floor. What it does enforce is that every query still
runs — the once-pass exits non-zero if any of them stops working, which is the
part worth failing CI over.

Was `bench/coverage.sh`. The report parsing in particular was two `awk`
one-liners indexing `$8`/`$9` out of llvm-cov's table with nothing explaining
where those numbers came from; here the column layout is named once and covered
by tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# `llvm-cov report` emits a fixed, whitespace-separated table:
#
#   Filename Regions Missed_Regions Cover Functions Missed_Functions Executed \
#   Lines Missed_Lines Cover Branches ...
#
# so index 7 is total lines and index 8 is missed lines. The shell version
# hardcoded these as awk's $8/$9 with no note of what they were.
_LINES = 7
_MISSED_LINES = 8

# The FFI bindings are generated, and vendored/toolchain sources are not ours.
IGNORE_RE = r"(GraphBLAS\.rs|graphblas/mod\.rs|\.cargo|rustc)"


@dataclass
class FileCoverage:
    path: str
    lines: int
    missed: int

    @property
    def covered(self) -> int:
        return self.lines - self.missed

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.lines if self.lines else 0.0


@dataclass
class Coverage:
    files: list[FileCoverage]

    @property
    def lines(self) -> int:
        return sum(f.lines for f in self.files)

    @property
    def missed(self) -> int:
        return sum(f.missed for f in self.files)

    @property
    def covered(self) -> int:
        return self.lines - self.missed

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.lines if self.lines else 0.0

    def least_covered(self, min_lines: int = 200, limit: int = 15) -> list[FileCoverage]:
        big = [f for f in self.files if f.lines > min_lines]
        return sorted(big, key=lambda f: f.percent)[:limit]


def parse_report(text: str, prefix: str = "graph/src") -> Coverage:
    """Pull per-file line counts for `prefix` out of an `llvm-cov report` table.

    Rows whose filename does not contain `prefix` are skipped, as are the TOTAL
    row and any row too short to carry line columns — llvm-cov wraps long paths
    onto their own line, which would otherwise be read as a data row.
    """
    files: list[FileCoverage] = []
    for line in text.splitlines():
        if prefix not in line:
            continue
        parts = line.split()
        if len(parts) <= _MISSED_LINES:
            continue
        try:
            lines, missed = int(parts[_LINES]), int(parts[_MISSED_LINES])
        except ValueError:
            continue
        files.append(FileCoverage(path=parts[0], lines=lines, missed=missed))
    return Coverage(files=files)


def _llvm_tool(name: str) -> Path:
    """Locate an llvm-tools binary belonging to the *rust toolchain*.

    The toolchain's copy is required, not merely preferred: LLVM's instrumentation
    profile format is versioned and coupled to the rustc that emitted the
    `.profraw`. A system LLVM (Homebrew's, say) is usually a different major
    version and fails with "unsupported instrumentation profile format version".
    So `~/.rustup/toolchains` is searched first and `PATH` is only a fallback —
    getting this backwards is easy, because on a dev machine `shutil.which` finds
    a perfectly real llvm-profdata that cannot read these profiles.

    These binaries are not on PATH by default: they ship in the
    `llvm-tools-preview` component, which is not installed by default. The shell
    version globbed `~/.rustup` and, when it found nothing, silently built the
    path `./llvm-profdata` and failed with a confusing "No such file or
    directory" — hence the explicit error below.
    """
    roots = [Path.home() / ".rustup/toolchains"]
    rustc = shutil.which("rustc")
    if rustc:
        # rustup shims resolve to ~/.rustup/toolchains/<tc>/bin/rustc, so the
        # sibling lib/ tree is where a non-default RUSTUP_HOME keeps them.
        roots.append(Path(rustc).resolve().parent.parent / "lib")
    for root in roots:
        if not root.exists():
            continue
        for found in sorted(root.rglob(name)):
            if found.is_file() and os.access(found, os.X_OK):
                return found

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise RuntimeError(
        f"{name} not found. Install it with: rustup component add llvm-tools-preview"
    )


def module_extension() -> str:
    return "dylib" if sys.platform == "darwin" else "so"


def build_flags() -> tuple[str, dict[str, str]]:
    """RUSTFLAGS and extra env for an instrumented build."""
    flags = "-C instrument-coverage"
    env: dict[str, str] = {}
    if sys.platform != "darwin":
        # Required on Linux (and so in CI): the embedded RediSearch static libs
        # otherwise fail to link with duplicate-symbol errors. macOS's linker
        # neither takes the flag nor needs it.
        flags += " -C link-arg=-Wl,--allow-multiple-definition"
        # graph/build.rs compiles C++ shims; the toolchain image has no default.
        env["CXX"] = os.environ.get("CXX", "clang++")
    return flags, env


def run(root: Path, bench_dir: Path, *, port: int, echo=print) -> Coverage:
    """The whole loop: instrumented build, one pass over the query set, report."""
    covdir = bench_dir / "results/cov"
    shutil.rmtree(covdir, ignore_errors=True)
    covdir.mkdir(parents=True)

    flags, extra_env = build_flags()
    env = {**os.environ, "RUSTFLAGS": flags, **extra_env}

    echo("== instrumented debug build ==")
    subprocess.run(["cargo", "build"], cwd=root, env=env, check=True)

    module = root / f"target/debug/libfalkordb.{module_extension()}"
    if not module.exists():
        raise RuntimeError(f"instrumented module not found at {module}")

    echo("== running the query set once each ==")
    # A subprocess, deliberately: the instrumented server inherits
    # LLVM_PROFILE_FILE from it, and keeping the measured run in its own process
    # means this command's own imports never land in the profile.
    once_env = {**os.environ, "LLVM_PROFILE_FILE": str(covdir / "cov-%p.profraw")}
    once = subprocess.run(
        [
            sys.executable,
            "-m",
            "falkorbench.cli",
            "measure",
            "--once",
            "--port",
            str(port),
            "--module",
            str(module),
        ],
        cwd=root,
        env=once_env,
    )

    profraws = sorted(covdir.glob("*.profraw"))
    if not profraws:
        raise RuntimeError(
            "no .profraw written — the instrumented server never flushed. It must "
            "be shut down gracefully (SHUTDOWN NOSAVE), not killed."
        )

    profdata = covdir / "cov.profdata"
    subprocess.run(
        [
            str(_llvm_tool("llvm-profdata")),
            "merge",
            "--sparse",
            *map(str, profraws),
            "-o",
            str(profdata),
        ],
        check=True,
    )
    report = subprocess.run(
        [
            str(_llvm_tool("llvm-cov")),
            "report",
            "--instr-profile",
            str(profdata),
            str(module),
            f"--ignore-filename-regex={IGNORE_RE}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (covdir / "report.txt").write_text(report)

    cov = parse_report(report)
    echo("")
    echo("== graph crate coverage (excluding the generated GraphBLAS FFI) ==")
    echo(f"lines: {cov.covered}/{cov.lines} = {cov.percent:.1f}%")
    echo("")
    echo("== least-covered graph/src files (>200 lines) ==")
    for f in cov.least_covered():
        echo(f"{f.percent:7.1f}%  {f.lines:6d} lines  {f.path}")
    echo("")
    echo(f"full report: {covdir / 'report.txt'}")

    # The once-pass exit code is the part that matters for CI: a query that
    # stopped working is a real failure, whereas the percentage is informational.
    if once.returncode != 0:
        raise RuntimeError(
            f"the query set did not run clean (exit {once.returncode}) — see the "
            f"FAIL lines above. Coverage numbers above are still valid."
        )
    return cov
