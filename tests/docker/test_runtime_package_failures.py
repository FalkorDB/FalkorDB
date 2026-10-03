"""Exercise Debian runtime package installation without touching host packages."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
STUB = '''#!/bin/sh
command_name=${0##*/}
printf '%s:%s\n' "$command_name" "$*" >> "$COMMAND_LOG"
case "$command_name:$*" in
    "$FAIL_COMMAND") exit 42 ;;
esac
'''


def package_run(path):
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if line.startswith("RUN apt-get update"):
            parts = [line.removeprefix("RUN ").rstrip("\\")]
            for line in lines:
                if line.lstrip().startswith("#"):
                    continue
                parts.append(line.rstrip("\\"))
                if not line.endswith("\\"):
                    return "\n".join(parts)
    raise AssertionError(f"No runtime package installation found in {path}")


class RuntimePackageFailures(unittest.TestCase):
    def test_package_failures(self):
        cases = {
            "": True,
            "apt-get:update": False,
            "apt-get:install -y --no-install-recommends libgomp1 ca-certificates bash gosu uidmap": False,
            "apt-get:dist-upgrade -y": False,
            "apt-get:autoremove -y": False,
            "apt-get:purge -y python3.13 python3.13-minimal libpython3.13-minimal libpython3.13-stdlib": True,
            "dpkg:--force-remove-essential --purge ncurses-base ncurses-bin": True,
            "apt-get:autoremove -y --purge": True,
        }
        for filename in ("Dockerfile", "Dockerfile.server"):
            script = package_run(ROOT / "build" / "docker" / filename)
            for failure, success in cases.items():
                if filename == "Dockerfile.server":
                    failure = failure.removesuffix(" uidmap")
                with self.subTest(dockerfile=filename, failure=failure):
                    with tempfile.TemporaryDirectory() as directory:
                        directory = Path(directory)
                        log = directory / "commands.log"
                        # Restrict PATH to inert commands: never invoke host APT,
                        # account management or filesystem cleanup commands.
                        for command in ("apt-get", "dpkg", "ln", "groupadd", "useradd", "rm", "gosu"):
                            stub = directory / command
                            stub.write_text(STUB)
                            stub.chmod(0o700)
                        result = subprocess.run(
                            ["/bin/sh", "-c", script],
                            env={"PATH": str(directory), "COMMAND_LOG": str(log),
                                 "FAIL_COMMAND": failure},
                            capture_output=True, text=True, timeout=5,
                        )
                        self.assertEqual(result.returncode, 0 if success else 42, result.stderr)
                        commands = log.read_text().splitlines()
                        if failure:
                            self.assertIn(failure, commands)
                        self.assertEqual("apt-get:clean" in commands, success)


if __name__ == "__main__":
    unittest.main()
