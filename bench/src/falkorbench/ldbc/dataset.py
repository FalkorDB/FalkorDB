"""Fetching and preparing the LDBC SNB Interactive v1 dataset.

Two preparation steps happen before anything is loaded, both of them
unavoidable rather than stylistic:

* **Ambiguous headers.** A self-referencing edge file is headed
  `Person.id|Person.id`. Parsed into a per-row map, the second column shadows
  the first — both endpoints resolve to the same node and every `KNOWS` edge
  becomes a self-loop. The header is rewritten to `FromPerson.id|ToPerson.id`.
* **Derived birthday parts.** IC10 needs the month and day of a person's
  birthday and FalkorDB has no `datetime()`, so two columns are appended.

Both are done by rewriting the file once, in Python. Files are rewritten in
place and a marker file records that it happened, so a re-run is a no-op
rather than double-appending columns.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
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

    cache.mkdir(parents=True, exist_ok=True)
    # The cache itself is the server's IMPORT_FOLDER (bind-mounted at
    # /import), so the server's user must be able to search it even under a
    # umask of 077. Only group/other read and search are added, also on a
    # cache the caller supplied; nothing else about it changes.
    try:
        cache.chmod(cache.stat().st_mode | 0o055)
    except OSError as e:
        raise DatasetError(f"could not make the cache {cache} readable: {e}") from e
    target = dataset_dir(cache, sf)
    if target.is_dir():
        echo(f"dataset already extracted: {target}")
        return target

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
    # Into a scratch directory first, renamed into place only once complete:
    # `target` existing is what marks the dataset as extracted, so a run
    # stopped mid-extraction must not leave a partial tree under that name.
    scratch = cache / f".extracting-{target.name}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    try:
        _extract_zst(tarball, scratch)
        if not (scratch / target.name).is_dir():
            raise DatasetError(f"{tarball.name} did not contain {target.name}")
        _readable_by_all(scratch / target.name)
        (scratch / target.name).rename(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


def _readable_by_all(root: Path) -> None:
    """Directories 0755, files readable by everyone, whatever the umask.

    The server that reads the dataset runs as another user (uid 999 in the
    container). Under a umask such as 077, `tarfile` creates directories
    0700 even with `filter="data"`, so the tree is opened up before it is
    published; nothing gains write or execute it did not have.
    """
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            os.chmod(dirpath, 0o755)
            for name in filenames:
                path = os.path.join(dirpath, name)
                os.chmod(path, (os.stat(path).st_mode & 0o7755) | 0o444)
    except OSError as e:
        raise DatasetError(f"could not make {root} readable: {e}") from e


#: Seconds a download may go without receiving a byte before it fails.
_DOWNLOAD_STALL_S = 120


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`.

    An explicit User-Agent is required, not cosmetic: datasets.ldbcouncil.org
    sits behind Cloudflare, which answers urllib's default
    `Python-urllib/3.x` with 403 while serving the same URL to curl.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "falkorbench-ldbc/1.0"})
    with (
        urllib.request.urlopen(request, timeout=_DOWNLOAD_STALL_S) as response,
        dest.open("wb") as fh,
    ):
        shutil.copyfileobj(response, fh)


def _extract_zst(tarball: Path, dest: Path) -> None:
    """Extract a .tar.zst.

    Python's tarfile has no zstd support before 3.14, so decompression goes
    through the `zstd` binary and only the tar step is done in-process.
    """
    if not hasattr(tarfile, "data_filter"):
        # The data filter arrived in Python 3.11.4 (Debian 12 ships 3.11.2).
        # Without it an archive could write outside the cache; stop instead.
        raise DatasetError("extracting the LDBC dataset safely needs Python 3.11.4 or newer")
    if not shutil.which("zstd"):
        raise DatasetError("`zstd` not found on PATH; install zstd to extract the dataset")
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            ["zstd", "--decompress", "--stdout", str(tarball)],
            stdout=subprocess.PIPE,
        )
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            # `filter="data"` refuses absolute paths, ../ escapes and links that escape,
            # and strips setuid and group/other write.
            tar.extractall(dest, filter="data")
        if proc.wait() != 0:
            raise DatasetError(f"zstd exited {proc.returncode}")
    except (OSError, tarfile.TarError) as e:
        raise DatasetError(f"extracting {tarball.name} failed: {e}") from e
    finally:
        # A refusal or a read error stops reading mid-stream; don't leave
        # zstd blocked on a full pipe.
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()


def prepare(root: Path, *, echo: Echo = print) -> None:
    """Rewrite ambiguous edge headers and derive the birthday columns.

    Idempotent: a marker file records completion, and a prepared tree is left
    untouched on a second call.
    """
    marker = root / _MARKER
    if marker.exists() and marker.read_text().strip() == _MARKER_VERSION:
        echo("dataset already prepared")
        return

    try:
        for edge in schema.EDGE_FILES:
            if edge.from_label == edge.to_label:
                _rewrite_header(root / edge.file, [edge.from_id, edge.to_id])

        _derive_birthday_parts(root / "dynamic/person_0_0.csv")
        marker.write_text(_MARKER_VERSION + "\n")
    except OSError as e:
        # A full disk or a permission problem is a dataset that is not ready,
        # reported like every other one. The per-file guards make a retry
        # pick up where this stopped.
        raise DatasetError(f"preparing {root.name} failed: {e}") from e
    echo(f"prepared {root.name}")


def _rewrite_header(path: Path, names: list[str]) -> None:
    """Replace the first `len(names)` header fields of `path`.

    The remaining fields (edge properties such as `creationDate`) are kept as
    they are. Rewriting only the header line means the file is not re-encoded,
    which matters at SF1 where these files run to hundreds of MB.
    """
    if not path.exists():
        raise DatasetError(f"missing dataset file: {path}")

    # Binary, so `tell()` is a byte offset: a text-mode `tell()` is an opaque
    # cookie, not a position the binary copy below may seek to.
    with path.open("rb") as fh:
        raw = fh.readline()
        if not raw:
            raise DatasetError(f"empty dataset file: {path}")
        rest_offset = fh.tell()
    header = raw.decode("utf-8")

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
    shutil.copymode(path, tmp)  # not the umask's: the server reads it as another user
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
    shutil.copymode(path, tmp)  # not the umask's: the server reads it as another user
    tmp.replace(path)
