#!/usr/bin/env python3
"""FalkorDB's OpenVEX document, and the checks that keep it true.

.vex/statements.toml is the source: one entry per CVE and OS, with the claim,
the reason a customer reads, and the evidence behind it. This script turns it
into the published documents, one per image (.vex/<image>.openvex.json), and
checks a document against a Trivy scan and the image filesystem the scan came
from.

Each statement's products are the packages themselves (pkg:deb/debian/libc6),
and the document is what scopes them to one image. Trivy (0.69) only applies a
statement scoped to the image, with the package as a subcomponent, when every
dependency path from the package up to the image passes through it, and
Debian's dependency cycles (libc6 <-> libgcc-s1) defeat that; a package as the
product it applies directly, in a registry scan and a tarball scan alike.

  vex.py generate              write .vex/<image>.openvex.json for every image
  vex.py generate --check      fail if a committed document is not what generate writes
  vex.py check --report SCAN.json --fs IMAGE.tar --image NAME --os OS
                               SCAN.json: `trivy image --vex .vex/NAME.openvex.json
                               --show-suppressed --format json`; IMAGE.tar:
                               `docker export` of the same image

`check` fails when
  - a HIGH or CRITICAL finding with no fixed version has no statement,
  - a finding hidden by a statement now has a fixed version (rebuild instead),
  - a statement hid nothing, or names a package Trivy did not apply it to,
  - a statement's evidence no longer holds in the image.
Findings that have a fixed version are left to the scan's own fixable gate.

Standard library only (Python 3.11+), so it runs on a stock CI runner.
"""

import argparse
import fnmatch
import json
import posixpath
import sqlite3
import stat
import struct
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VEX_DIR = ROOT / ".vex"
SOURCE = VEX_DIR / "statements.toml"


def document_path(image, out_dir=None):
    return Path(out_dir or VEX_DIR) / f"{image}.openvex.json"

STATUSES = {"not_affected", "affected", "under_investigation"}
JUSTIFICATIONS = {
    "component_not_present",
    "vulnerable_code_not_present",
    "vulnerable_code_not_in_execute_path",
    "vulnerable_code_cannot_be_controlled_by_adversary",
    "inline_mitigations_already_exist",
}
# check name -> (the keys it needs, the keys it may also take)
CHECKS = {
    "absent": ({"paths"}, set()),
    "contains": ({"path", "text"}, set()),
    "lacks": ({"path", "text"}, set()),
    "no-import": ({"symbols"}, {"except"}),
    "not-run": ({"packages"}, set()),
    "no-setid": (set(), set()),
    "version": ({"package", "is"}, set()),
}
# The tar header a `docker export` keeps a file's capabilities in.
CAPABILITY_XATTR = "SCHILY.xattr.security.capability"
STATEMENT_KEYS = {
    "cve", "os", "images", "packages", "status", "justification", "impact",
    "action", "status_notes", "evidence", "notes",
}


class VexError(Exception):
    pass


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

def load_source(path=None):
    path = path or SOURCE
    with open(path, "rb") as f:
        src = tomllib.load(f)
    errors = validate(src)
    if errors:
        raise VexError(f"{path}:\n  " + "\n  ".join(errors))
    return src


def validate(src):
    errors = []
    doc = src.get("document", {})
    for key, kind in (("id_prefix", str), ("author", str), ("timestamp", str), ("version", int)):
        if not isinstance(doc.get(key), kind):
            errors.append(f"[document] needs {key} ({kind.__name__})")
    images = src.get("images", {})
    oses = src.get("os", {})
    if not images or not oses:
        errors.append("[images] and [os] must both be present")
    for name, osdef in oses.items():
        if "{name}" not in osdef.get("purl", ""):
            errors.append(f"[os.{name}] purl must contain {{name}}")
    seen = set()
    for i, st in enumerate(src.get("statement", [])):
        where = f"statement {i + 1} ({st.get('cve', '?')}, {st.get('os', '?')})"
        unknown = set(st) - STATEMENT_KEYS
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
        for key in ("cve", "os", "status"):
            if not isinstance(st.get(key), str):
                errors.append(f"{where}: needs {key}")
        if st.get("os") not in oses:
            errors.append(f"{where}: os must be one of {sorted(oses)}")
        imgs = st.get("images")
        if not imgs or not set(imgs) <= set(images):
            errors.append(f"{where}: images must be a non-empty subset of {sorted(images)}")
        pkgs = st.get("packages")
        if not pkgs or len(set(pkgs)) != len(pkgs):
            errors.append(f"{where}: packages must be non-empty and unique")
        status = st.get("status")
        if status not in STATUSES:
            errors.append(f"{where}: status must be one of {sorted(STATUSES)}")
        if status == "not_affected":
            if st.get("justification") not in JUSTIFICATIONS:
                errors.append(f"{where}: not_affected needs a justification from {sorted(JUSTIFICATIONS)}")
            if not st.get("impact", "").strip():
                errors.append(f"{where}: not_affected needs an impact statement")
        elif "justification" in st or "impact" in st:
            errors.append(f"{where}: justification and impact are for not_affected only")
        if status == "affected" and not st.get("action", "").strip():
            errors.append(f"{where}: affected needs an action statement")
        for ev in st.get("evidence", []):
            check = ev.get("check")
            if check not in CHECKS:
                errors.append(f"{where}: unknown evidence check {check!r}")
            else:
                needs, may = CHECKS[check]
                keys = set(ev) - {"check"}
                if not needs <= keys <= needs | may:
                    errors.append(f"{where}: check {check!r} needs {sorted(needs)}"
                                  + (f" and may take {sorted(may)}" if may else ""))
        for img in imgs or []:
            key = (st.get("cve"), st.get("os"), img)
            if key in seen:
                errors.append(f"{where}: a second statement for {key}")
            seen.add(key)
    return errors


def applies(st, image, os_name):
    return st["os"] == os_name and image in st["images"]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def package_purl(src, os_name, package):
    return src["os"][os_name]["purl"].format(name=package)


def openvex_statement(src, st):
    out = {
        "vulnerability": {"name": st["cve"]},
        "products": [{"@id": package_purl(src, st["os"], p)} for p in sorted(st["packages"])],
        "status": st["status"],
    }
    if st["status"] == "not_affected":
        out["justification"] = st["justification"]
        out["impact_statement"] = " ".join(st["impact"].split())
    if st.get("action"):
        out["action_statement"] = " ".join(st["action"].split())
    if st.get("status_notes"):
        out["status_notes"] = " ".join(st["status_notes"].split())
    return out


def build_document(src, image):
    """The OpenVEX document for one image: every statement that lists it."""
    doc = src["document"]
    statements = sorted((st for st in src.get("statement", []) if image in st["images"]),
                        key=lambda st: (st["cve"], st["os"]))
    return {
        "@context": "https://openvex.dev/ns/v0.2.0",
        "@id": doc["id_prefix"] + document_path(image).name,
        "author": doc["author"],
        "timestamp": doc["timestamp"],
        "version": doc["version"],
        "tooling": "build/vex/vex.py, from .vex/statements.toml",
        "statements": [openvex_statement(src, st) for st in statements],
    }


def render(doc):
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def cmd_generate(args):
    src = load_source(args.source)
    stale = []
    for image in sorted(src["images"]):
        path, text = document_path(image, args.out_dir), render(build_document(src, image))
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path))
        else:
            path.write_text(text)
            print(f"wrote {path} ({len(json.loads(text)['statements'])} statements)")
    if stale:
        print(f"{', '.join(stale)} not what {SOURCE.relative_to(ROOT)} generates. "
              "Run: python3 build/vex/vex.py generate", file=sys.stderr)
        return 1
    if args.check:
        print("the VEX documents are up to date")
    return 0


# ---------------------------------------------------------------------------
# Image filesystem (a `docker export` tarball)
# ---------------------------------------------------------------------------

def member_path(name):
    """A tar member name as an absolute path."""
    return posixpath.normpath("/" + name.removeprefix("./").lstrip("/"))


class ImageFS:
    def __init__(self, tar_path):
        self.tar = tarfile.open(tar_path)
        self.members = {member_path(m.name): m for m in self.tar.getmembers()}

    def resolve(self, path, depth=0):
        """The path with every symlink followed, or None when it does not exist."""
        if depth > 40:
            return None
        parts = [p for p in posixpath.normpath(path).split("/") if p]
        cur = "/"
        for i, part in enumerate(parts):
            nxt = posixpath.join(cur, part)
            m = self.members.get(nxt)
            if m is None:
                return None
            if m.issym():
                target = m.linkname if m.linkname.startswith("/") else posixpath.join(cur, m.linkname)
                rest = "/".join(parts[i + 1:])
                return self.resolve(posixpath.join(target, rest) if rest else target, depth + 1)
            cur = nxt
        return cur

    def canonical(self, path):
        """resolve(), plus a hardlink mapped to the file it links to."""
        real = self.resolve(path)
        if real is None:
            return None
        m = self.members.get(real)
        if m is not None and m.islnk():
            return self.canonical(member_path(m.linkname))
        return real

    def owned(self, path):
        """The file a package's listed path really is, if it is file content:
        None for a symlink (pointing at some other package's file is not
        shipping it) and for a directory. Directory symlinks on the way are
        followed, so /lib/x.so finds /usr/lib/x.so."""
        parent = self.resolve(posixpath.dirname(path))
        if parent is None:
            return None
        m = self.members.get(posixpath.join(parent, posixpath.basename(path)))
        if m is None or m.issym() or m.isdir():
            return None
        return self.canonical(member_path(m.linkname)) if m.islnk() else member_path(m.name)

    def member(self, real):
        return self.members.get(real)

    def is_file(self, real):
        m = self.members.get(real)
        return m is not None and (m.isfile() or m.islnk())

    def open(self, real):
        return self.tar.extractfile(self.members[real])

    def files(self):
        for path, m in self.members.items():
            if m.isfile():
                yield path, m

    def read_text(self, path):
        real = self.canonical(path)
        return self.open(real).read().decode("utf-8", "replace") if real else None


# ---------------------------------------------------------------------------
# ELF (64-bit little-endian: x86_64 and aarch64)
# ---------------------------------------------------------------------------

class UnreadableElf(Exception):
    pass


class Elf:
    __slots__ = ("machine", "interp", "needed", "runpath", "imports", "exports")


def parse_elf(f):
    head = f.read(64)
    if len(head) < 64 or head[:4] != b"\x7fELF":
        return None
    if head[4] != 2 or head[5] != 1:
        raise UnreadableElf("not 64-bit little-endian")
    e = Elf()
    e.machine = struct.unpack_from("<H", head, 0x12)[0]
    phoff, shoff = struct.unpack_from("<QQ", head, 0x20)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", head, 0x36)

    def read(off, size):
        f.seek(off)
        return f.read(size)

    e.interp = None
    ph = read(phoff, phentsize * phnum)
    for i in range(phnum):
        p_type, _, p_offset, _, _, p_filesz = struct.unpack_from("<IIQQQQ", ph, i * phentsize)
        if p_type == 3:  # PT_INTERP
            e.interp = read(p_offset, p_filesz).rstrip(b"\0").decode()
    e.needed, e.runpath, e.imports, e.exports = [], [], set(), set()
    if shnum == 0:
        raise UnreadableElf("no section headers")
    sh = read(shoff, shentsize * shnum)
    sections = [struct.unpack_from("<IIQQQQIIQQ", sh, i * shentsize) for i in range(shnum)]

    def strtab(index):
        _, _, _, _, off, size, *_ = sections[index]
        return read(off, size)

    def string(tab, off):
        return tab[off:tab.index(b"\0", off)].decode("utf-8", "replace")

    for _, sh_type, _, _, off, size, link, _, _, entsize in sections:
        if sh_type == 6:  # SHT_DYNAMIC
            strs, data = strtab(link), read(off, size)
            for j in range(0, len(data), 16):
                tag, val = struct.unpack_from("<qQ", data, j)
                if tag == 0:
                    break
                if tag == 1:  # DT_NEEDED
                    e.needed.append(string(strs, val))
                elif tag in (15, 29):  # DT_RPATH, DT_RUNPATH
                    e.runpath += string(strs, val).split(":")
        elif sh_type == 11:  # SHT_DYNSYM
            strs, data = strtab(link), read(off, size)
            for j in range(entsize, len(data), entsize):  # entry 0 is the null symbol
                name_off, _, _, shndx = struct.unpack_from("<IBBH", data, j)
                name = string(strs, name_off)
                if name:
                    (e.imports if shndx == 0 else e.exports).add(name)
    return e


TRIPLETS = {62: "x86_64-linux-gnu", 183: "aarch64-linux-gnu"}
# Where `#!/usr/bin/env NAME` finds NAME: the PATH every one of our images sets.
PATH_DIRS = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")


class Analysis:
    """ELF facts about one image, computed on demand and cached."""

    def __init__(self, fs):
        self.fs = fs
        self._elf = {}
        self._libdirs = None
        self.unreadable = {}

    def elf(self, real):
        if real not in self._elf:
            e = None
            if self.fs.is_file(real):
                try:
                    e = parse_elf(self.fs.open(real))
                except (UnreadableElf, struct.error, ValueError, IndexError) as exc:
                    self.unreadable[real] = str(exc)
            self._elf[real] = e
        return self._elf[real]

    def libdirs(self, machine):
        if self._libdirs is None:
            dirs = []

            def conf(path):
                text = self.fs.read_text(path) or ""
                for line in text.splitlines():
                    line = line.split("#")[0].strip()
                    if line.startswith("include "):
                        pattern = line.split(None, 1)[1]
                        for p in sorted(self.fs.members):
                            if fnmatch.fnmatch(p, pattern):
                                conf(p)
                    elif line:
                        dirs.append(line)

            conf("/etc/ld.so.conf")
            self._libdirs = dirs
        t = TRIPLETS.get(machine, "")
        return self._libdirs + [f"/lib/{t}", f"/usr/lib/{t}", "/lib64", "/usr/lib64", "/lib", "/usr/lib"]

    def find_library(self, name, origin_dir, elf):
        dirs = [d.replace("${ORIGIN}", origin_dir).replace("$ORIGIN", origin_dir) for d in elf.runpath]
        for d in dirs + self.libdirs(elf.machine):
            real = self.fs.canonical(posixpath.join(d, name))
            if real and self.fs.is_file(real):
                return real
        return None

    def loadable(self, real, machine):
        """Whether this ELF file can load here: built for the image's machine,
        with every DT_NEEDED found. The browser ships native addons for other
        platforms (x86-64 on arm64, musl builds) that never load."""
        e = self.elf(real)
        return (e is not None and e.machine == machine
                and all(self.find_library(n, posixpath.dirname(real), e) for n in e.needed))

    def interpreters(self, real):
        """The programs a script's #! line runs: its interpreter and, for
        `#!/usr/bin/env NAME`, NAME. On RHEL and Amazon Linux, chown, cat and
        the other coreutils tools are such scripts, run by /usr/bin/coreutils."""
        if not self.fs.is_file(real):
            return []
        line = self.fs.open(real).read(256).split(b"\n")[0]
        words = line[2:].decode("utf-8", "replace").split() if line.startswith(b"#!") else []
        if not words:
            return []
        if posixpath.basename(words[0]) == "env" and len(words) > 1 and not words[1].startswith("-"):
            name = words[1]
            found = [p for p in (posixpath.join(d, name) for d in PATH_DIRS) if self.fs.canonical(p)]
            return [words[0], found[0] if found else name]
        return [words[0]]

    def closure(self, roots):
        """Every file the roots run or load through #!, PT_INTERP and DT_NEEDED,
        roots included."""
        seen, todo, missing = set(), list(roots), []
        while todo:
            real = todo.pop()
            if real in seen:
                continue
            seen.add(real)
            e = self.elf(real)
            if e is None and real in self.unreadable:
                # Its DT_NEEDED would be lost, and not-run evidence pass unchecked.
                missing.append(f"{real} is an ELF file this checker cannot read ({self.unreadable[real]})")
                continue
            if e is None:
                for prog in self.interpreters(real):
                    interp = self.fs.canonical(prog)
                    if interp is None:
                        missing.append(f"{real} runs {prog}")
                    else:
                        todo.append(interp)
                continue
            deps = [self.fs.canonical(e.interp)] if e.interp else []
            for name in e.needed:
                lib = self.find_library(name, posixpath.dirname(real), e)
                if lib is None:
                    missing.append(f"{real} needs {name}")
                deps.append(lib)
            todo += [d for d in deps if d]
        return seen, missing

    def elf_files(self):
        for path, _ in self.fs.files():
            e = self.elf(path)
            if e is not None:
                yield path, e


# ---------------------------------------------------------------------------
# Package databases, read from the filesystem (no need to run the image,
# so an arm64 image is checked on an amd64 runner without emulation)
# ---------------------------------------------------------------------------

def dpkg_files(fs):
    out = {}
    for path, m in fs.files():
        if path.startswith("/var/lib/dpkg/info/") and path.endswith(".list"):
            pkg = posixpath.basename(path)[:-5].split(":")[0]
            out.setdefault(pkg, []).extend(
                l for l in fs.open(path).read().decode().splitlines() if l and l != "/.")
    return out


def rpm_files(fs):
    real = fs.canonical("/var/lib/rpm/rpmdb.sqlite")
    if real is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        tmp.write(fs.open(real).read())
        tmp.flush()
        db = sqlite3.connect(tmp.name)
        out = {}
        for (blob,) in db.execute("SELECT blob FROM Packages"):
            tags = rpm_header(blob)
            names = tags.get(1000, [])
            if not names:
                continue
            base, dirs, idx = tags.get(1117, []), tags.get(1118, []), tags.get(1116, [])
            out.setdefault(names[0], []).extend(dirs[i] + b for i, b in zip(idx, base))
        db.close()
    return out


def rpm_header(blob):
    """tag -> list of values, for the string, string-array and int32 tags."""
    il, dl = struct.unpack_from(">II", blob, 0)
    store = 8 + 16 * il
    tags = {}
    for i in range(il):
        tag, typ, off, count = struct.unpack_from(">IIiI", blob, 8 + 16 * i)
        if typ in (6, 8, 9):  # STRING, STRING_ARRAY, I18NSTRING
            vals, pos = [], store + off
            for _ in range(count):
                end = blob.index(b"\0", pos)
                vals.append(blob[pos:end].decode("utf-8", "replace"))
                pos = end + 1
            tags[tag] = vals
        elif typ == 4:  # INT32
            tags[tag] = list(struct.unpack_from(f">{count}i", blob, store + off))
    return tags


def package_files(fs):
    files = rpm_files(fs)
    if files is None:
        files = dpkg_files(fs)
    if not files:
        raise VexError("no dpkg or rpm package database in the image")
    return files


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def runtime_roots(src, fs, image, os_name, errors):
    """The files the container runs or loads directly, per [runtime]."""
    rt = src.get("runtime", {})
    os_rt = rt.get("os", {}).get(os_name, {})
    programs = list(os_rt.get("programs", []))
    dlopen = list(os_rt.get("dlopen", []))
    trees = []
    for section in (rt.get("image", {}).get(image, {}),):
        programs += section.get("programs", [])
        trees += section.get("trees", [])
    roots = set()
    for p in programs:
        real = fs.canonical(p)
        if real is None:
            errors.append(f"[runtime] lists {p}, which is not in the {image} image: update .vex/statements.toml")
        else:
            roots.add(real)
    if trees or dlopen:
        an = Analysis(fs)
        machines = {e.machine for e in map(an.elf, roots) if e is not None}
        if len(machines) != 1:
            errors.append(f"cannot tell the image's machine type from the runtime programs ({machines})")
            return roots
        machine = machines.pop()

    def loadable_named(match):
        """The loadable files a matching name is, links followed: a symlink
        in a module directory can name a library anywhere."""
        found = set()
        for p in fs.members:
            if match(p):
                real = fs.canonical(p)
                if real and fs.is_file(real) and an.loadable(real, machine):
                    found.add(real)
        return found

    for t in trees:
        real = fs.resolve(t)
        if real is None:
            errors.append(f"[runtime] lists the tree {t}, which is not in the {image} image")
            continue
        roots |= loadable_named(lambda p: p.startswith(real + "/"))
    # Libraries the runtime loads by name: every one that matches counts,
    # whether or not this configuration loads it.
    for pattern in dlopen:
        roots |= loadable_named(lambda p: fnmatch.fnmatch(p, pattern))
    return roots


def load_report(path):
    with open(path) as f:
        report = json.load(f)
    remaining, hidden = [], []
    for result in report.get("Results") or []:
        remaining += result.get("Vulnerabilities") or []
        hidden += [m for m in result.get("ExperimentalModifiedFindings") or []
                   if m.get("Type") == "vulnerability"]
    return remaining, hidden


def cmd_check(args):
    src = load_source(args.source)
    if args.image not in src["images"] or args.os not in src["os"]:
        raise VexError(f"--image must be one of {sorted(src['images'])}, --os one of {sorted(src['os'])}")
    fail_on = {s.strip().upper() for s in args.fail_on.split(",") if s.strip()}
    statements = [st for st in src.get("statement", []) if applies(st, args.image, args.os)]
    by_cve = {st["cve"]: st for st in statements}
    remaining, hidden = load_report(args.report)
    errors, warnings = [], []

    # Findings Trivy still reports.
    for v in remaining:
        cve, pkg, sev = v["VulnerabilityID"], v["PkgName"], v.get("Severity", "UNKNOWN")
        if v.get("FixedVersion"):
            continue  # the fixable gate's business
        st = by_cve.get(cve)
        if st is None:
            msg = f"{cve} ({sev}) in {pkg} has no statement"
            (errors if sev in fail_on else warnings).append(msg)
        elif pkg not in st["packages"]:
            errors.append(f"the statement for {cve} does not name {pkg}, where Trivy also reports it")
        elif st["status"] == "not_affected":
            errors.append(f"Trivy did not apply the not_affected statement for {cve} to {pkg}: "
                          "check the product and package identifiers")

    # Findings a statement hid.
    hidden_by = {}
    for m in hidden:
        finding = m.get("Finding", {})
        cve, pkg = finding.get("VulnerabilityID"), finding.get("PkgName")
        hidden_by.setdefault(cve, set()).add(pkg)
        st = by_cve.get(cve)
        if st is None or st["status"] != "not_affected" or pkg not in st["packages"]:
            errors.append(f"{cve} in {pkg} is hidden, but no not_affected statement in statements.toml "
                          f"names {pkg} for it: the document Trivy applied does not match its source")
        if finding.get("FixedVersion"):
            errors.append(f"{cve} in {pkg} now has a fixed version ({finding['FixedVersion']}), "
                          "but a statement hides it: rebuild with the fix, then drop the statement")

    # Statements that matched nothing.
    reported = {}
    for v in remaining:
        reported.setdefault(v["VulnerabilityID"], set()).add(v["PkgName"])
    for st in statements:
        cve = st["cve"]
        # A not_affected statement's findings are hidden; any other status leaves them reported.
        matched = hidden_by if st["status"] == "not_affected" else reported
        if cve not in matched:
            if st["status"] == "not_affected":
                errors.append(f"the statement for {cve} hid nothing in this scan: Trivy no longer "
                              "reports it, or the statement does not match")
            else:
                errors.append(f"the {st['status']} statement for {cve} matches no finding in this scan")
            continue
        for pkg in sorted(set(st["packages"]) - matched[cve]):
            errors.append(f"the statement for {cve} names {pkg}, which this scan does not report it in: "
                          "drop the package, or split the statement by image")

    # Evidence.
    if args.fs:
        fs = ImageFS(args.fs)
        an = Analysis(fs)
        roots = runtime_roots(src, fs, args.image, args.os, errors)
        runtime, missing = an.closure(roots)
        errors += [f"cannot resolve what the runtime runs or loads: {m}" for m in missing]
        pkg_files = package_files(fs)
        versions = {}
        for v in remaining + [m.get("Finding", {}) for m in hidden]:
            versions.setdefault(v.get("PkgName"), set()).add(v.get("InstalledVersion"))
        imports = None

        def imports_index():
            nonlocal imports
            if imports is None:
                imports = {}
                for path, e in an.elf_files():
                    for sym in e.imports:
                        imports.setdefault(sym, []).append(path)
                if an.unreadable:
                    errors.append(f"cannot read the dynamic symbols of {sorted(an.unreadable)}, "
                                  "so no-import cannot be checked there")
            return imports

        for st in statements:
            for ev in st.get("evidence", []):
                problem = evaluate(ev, fs, an, runtime, pkg_files, versions, imports_index)
                if problem:
                    errors.append(f"{st['cve']}: evidence {json.dumps(ev)} no longer holds: {problem}")
    elif any(st.get("evidence") for st in statements):
        errors.append("statements carry evidence, but no --fs image was given to check it against")

    report_out(args, statements, remaining, hidden, errors, warnings)
    return 1 if errors else 0


def evaluate(ev, fs, an, runtime, pkg_files, versions, imports_index):
    check = ev["check"]
    if check == "absent":
        present = [p for p in ev["paths"] if fs.resolve(p) is not None]
        return f"present: {present}" if present else None
    if check in ("contains", "lacks"):
        real = fs.canonical(ev["path"])
        if real is None or not fs.is_file(real):
            return f"{ev['path']} is not in the image"
        found = ev["text"].encode() in fs.open(real).read()
        if check == "contains":
            return None if found else f"{ev['path']} does not contain {ev['text']!r}"
        return f"{ev['path']} contains {ev['text']!r}" if found else None
    if check == "no-setid":
        setid = sorted(p for p, m in fs.files()
                       if m.mode & (stat.S_ISUID | stat.S_ISGID) or CAPABILITY_XATTR in m.pax_headers)
        return f"setuid, setgid or capability-carrying files: {setid}" if setid else None
    if check == "no-import":
        index, allowed = imports_index(), {fs.canonical(p) for p in ev.get("except", [])}
        hits = {s: sorted(set(index[s]) - allowed) for s in ev["symbols"] if set(index.get(s, [])) - allowed}
        return f"imported by {hits}" if hits else None
    if check == "not-run":
        hits = {}
        for pkg in ev["packages"]:
            if pkg not in pkg_files:
                return f"package {pkg} is not installed (use a different check, or drop the package)"
            for f in pkg_files[pkg]:
                real = fs.owned(f)
                if real in runtime:
                    hits.setdefault(pkg, []).append(real)
        return f"run or loaded: {hits}" if hits else None
    if check == "version":
        have = versions.get(ev["package"])
        if not have:
            return f"the scan reports no version for {ev['package']}"
        return None if have == {ev["is"]} else f"installed {sorted(have)}, the statement was written for {ev['is']}"
    return f"unknown check {check}"


def report_out(args, statements, remaining, hidden, errors, warnings):
    lines = [f"VEX check: {args.image} ({args.os}), {len(statements)} statements, "
             f"{len(hidden)} findings hidden, {len(remaining)} still reported"]
    lines += [f"ERROR: {e}" for e in errors]
    lines += [f"warning: {w}" for w in warnings]
    print("\n".join(lines))
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(f"### VEX check: `{args.image}` ({args.os})\n\n")
            f.write(f"{len(statements)} statements; {len(hidden)} findings hidden; "
                    f"{len(remaining)} still reported.\n\n")
            for e in errors:
                f.write(f"- :x: {e}\n")
            for w in warnings:
                f.write(f"- :warning: {w}\n")
            if not errors and not warnings:
                f.write("Every statement matched, and its evidence holds.\n")
            f.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help=argparse.SUPPRESS)  # tests: another statements file
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--check", action="store_true")
    g.add_argument("--out-dir", help=argparse.SUPPRESS)  # tests: write somewhere else
    c = sub.add_parser("check")
    c.add_argument("--report", required=True)
    c.add_argument("--fs", help="`docker export` tarball of the scanned image")
    c.add_argument("--image", required=True)
    c.add_argument("--os", required=True)
    c.add_argument("--fail-on", default="HIGH,CRITICAL")
    c.add_argument("--summary", help="append a markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    args = p.parse_args(argv)
    try:
        return cmd_generate(args) if args.cmd == "generate" else cmd_check(args)
    except VexError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
