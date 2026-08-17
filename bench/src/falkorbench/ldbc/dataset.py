"""Fetching and preparing the LDBC SNB Interactive v1 dataset.

Two preparation steps happen before anything is loaded, both of them
unavoidable rather than stylistic:

* **Ambiguous headers.** A self-referencing edge file is headed
  `Person.id|Person.id`. Parsed into a per-row map, the second column shadows
  the first — both endpoints resolve to the same node and every `KNOWS` edge
  becomes a self-loop. The header is rewritten to `FromPerson.id|ToPerson.id`.
* **Derived birthday parts.** IC10 needs the month and day of a person's
  birthday and FalkorDB has no `datetime()`, so two columns are appended.

Both are done by rewriting the file once, in Python. The pre-existing
`tests/test_ldbc.py` used `sed -i "" ...` for the header rewrite, which is the
BSD/macOS spelling and exits 2 under GNU sed — so that path only ever worked on
one platform.

Files are rewritten in place and a marker file records that it happened, so a
re-run is a no-op rather than double-appending columns.
"""

from __future__ import annotations

import csv
import datetime as dt
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from falkorbench.ldbc import schema

#: Written into the dataset directory once preparation has succeeded. Its
#: contents are the marker version, so a change to the preparation logic here
#: invalidates previously prepared trees instead of silently reusing them.
_MARKER = ".falkorbench-prepared"
_MARKER_VERSION = "1"

Echo = Callable[[str], None]


class DatasetError(RuntimeError):
    """The dataset could not be fetched or prepared."""


def dataset_dir(cache: Path, sf: str) -> Path:
    return cache / schema.DATASET_DIR.format(sf=sf)


def fetch(cache: Path, sf: str, *, echo: Echo = print) -> Path:
    """Download and extract the SF `sf` dataset into `cache`, if not present.

    Returns the extracted dataset directory.
    """
    if sf not in schema.SCALE_FACTORS:
        raise DatasetError(f"unknown scale factor {sf!r}; known: {', '.join(schema.SCALE_FACTORS)}")

    target = dataset_dir(cache, sf)
    if target.is_dir():
        echo(f"dataset already extracted: {target}")
        return target

    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / schema.DATASET_TARBALL.format(sf=sf)
    expected = schema.SCALE_FACTORS[sf]

    if not tarball.exists() or tarball.stat().st_size != expected:
        url = f"{schema.DATASET_BASE_URL}/{tarball.name}"
        echo(f"downloading {url} ({expected / 1e6:.0f} MB)...")
        # .part so an interrupted download is never mistaken for a complete one.
        part = tarball.with_suffix(tarball.suffix + ".part")
        try:
            _download(url, part)
        except OSError as e:
            part.unlink(missing_ok=True)
            raise DatasetError(f"download failed: {e}") from e
        size = part.stat().st_size
        if size != expected:
            part.unlink(missing_ok=True)
            raise DatasetError(
                f"downloaded {size} bytes, expected {expected}; "
                f"the published dataset may have been republished"
            )
        part.rename(tarball)

    echo(f"extracting {tarball.name}...")
    _extract_zst(tarball, cache)
    if not target.is_dir():
        raise DatasetError(f"{tarball.name} did not contain {target.name}")
    return target


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`.

    An explicit User-Agent is required, not cosmetic: datasets.ldbcouncil.org
    sits behind Cloudflare, which answers urllib's default
    `Python-urllib/3.x` with 403 while serving the same URL to curl.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "falkorbench-ldbc/1.0"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def _extract_zst(tarball: Path, dest: Path) -> None:
    """Extract a .tar.zst.

    Python's tarfile has no zstd support before 3.14, so decompression goes
    through the `zstd` binary and only the tar step is done in-process.
    """
    if not shutil.which("zstd"):
        raise DatasetError("`zstd` not found on PATH; install zstd to extract the dataset")
    try:
        proc = subprocess.Popen(
            ["zstd", "--decompress", "--stdout", str(tarball)],
            stdout=subprocess.PIPE,
        )
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            # `filter="data"` refuses absolute paths and ../ escapes. Default in
            # 3.14; set explicitly so behaviour does not depend on the runtime.
            tar.extractall(dest, filter="data")
        if proc.wait() != 0:
            raise DatasetError(f"zstd exited {proc.returncode}")
    except (OSError, tarfile.TarError) as e:
        raise DatasetError(f"extracting {tarball.name} failed: {e}") from e


def prepare(root: Path, *, echo: Echo = print) -> None:
    """Rewrite ambiguous edge headers and derive the birthday columns.

    Idempotent: a marker file records completion, and a prepared tree is left
    untouched on a second call.
    """
    marker = root / _MARKER
    if marker.exists() and marker.read_text().strip() == _MARKER_VERSION:
        echo("dataset already prepared")
        return

    for edge in schema.EDGE_FILES:
        if edge.from_label == edge.to_label:
            _rewrite_header(root / edge.file, [edge.from_id, edge.to_id])

    _derive_birthday_parts(root / "dynamic/person_0_0.csv")
    marker.write_text(_MARKER_VERSION + "\n")
    echo(f"prepared {root.name}")


def _rewrite_header(path: Path, names: list[str]) -> None:
    """Replace the first `len(names)` header fields of `path`.

    The remaining fields (edge properties such as `creationDate`) are kept as
    they are. Rewriting only the header line means the file is not re-encoded,
    which matters at SF1 where these files run to hundreds of MB.
    """
    if not path.exists():
        raise DatasetError(f"missing dataset file: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline()
        if not header:
            raise DatasetError(f"empty dataset file: {path}")
        rest_offset = fh.tell()

    fields = header.rstrip("\r\n").split("|")
    if len(fields) < len(names):
        raise DatasetError(f"{path.name}: expected >= {len(names)} columns, got {fields}")
    if fields[: len(names)] == names:
        return  # already rewritten
    fields[: len(names)] = names
    new_header = "|".join(fields) + "\n"

    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open("rb") as src, tmp.open("wb") as dst:
        dst.write(new_header.encode("utf-8"))
        src.seek(rest_offset)
        shutil.copyfileobj(src, dst)
    tmp.replace(path)


def _derive_birthday_parts(path: Path) -> None:
    """Append `birthdayMonth` and `birthdayDay` columns to the person file.

    The `birthday` column is epoch milliseconds at UTC midnight. IC10 asks for
    the calendar month and day of that date, which FalkorDB cannot compute
    without a temporal type, so both are materialised here.
    """
    if not path.exists():
        raise DatasetError(f"missing dataset file: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("|")
    if "birthdayMonth" in header:
        return
    if "birthday" not in header:
        raise DatasetError(f"{path.name}: no `birthday` column in {header}")
    birthday_idx = header.index("birthday")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with (
        path.open("r", encoding="utf-8", newline="") as src,
        tmp.open("w", encoding="utf-8", newline="") as dst,
    ):
        reader = csv.reader(src, delimiter="|")
        writer = csv.writer(dst, delimiter="|", lineterminator="\n")
        writer.writerow([*next(reader), "birthdayMonth", "birthdayDay"])
        for row in reader:
            try:
                millis = int(row[birthday_idx])
            except (IndexError, ValueError) as e:
                raise DatasetError(f"{path.name}: bad birthday in row {row[:1]}: {e}") from e
            date = dt.datetime.fromtimestamp(millis / 1000, tz=dt.UTC)
            writer.writerow([*row, date.month, date.day])
    tmp.replace(path)
