import glob
import os

try:
    from .spec_loader import load_specs_file
    from .spec_types import QuerySpec
except ImportError:
    from spec_loader import load_specs_file
    from spec_types import QuerySpec


def discover_spec_files(specs_dir: str) -> list[str]:
    pattern = os.path.join(specs_dir, "spec_*.yml")
    return sorted(glob.glob(pattern))


def load_discovered_specs(specs_dir: str) -> list[tuple[str, list[QuerySpec]]]:
    files = discover_spec_files(specs_dir)
    return [(path, load_specs_file(path)) for path in files]
