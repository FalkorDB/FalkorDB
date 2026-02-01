import os
import platform
import sys
from functools import wraps

import redis
from base import FlowTestsBase
from falkordb import Edge, ExecutionPlan, FalkorDB, Graph, Node, Path
from redis import ResponseError
from RLTest import Defaults
from RLTest import Env as Environment

Defaults.decode_responses = True


# Platform detection (replaces paella dependency)
def _get_os():
    """Get OS name (linux, macos, etc.)"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system


def _get_arch():
    """Get architecture (x64, arm64v8, etc.)"""
    machine = platform.machine().lower()
    if machine == "x86_64" or machine == "amd64":
        return "x64"
    elif machine == "aarch64" or machine == "arm64":
        return "arm64v8"
    return machine


def _get_osnick():
    """Get OS nickname (e.g., ubuntu22.04, macos, etc.)"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    # Try to get Linux distribution info
    try:
        # Python 3.10+ has platform.freedesktop_os_release()
        if hasattr(platform, "freedesktop_os_release"):
            info = platform.freedesktop_os_release()
            dist_id = info.get("ID", "linux").lower()
            version_id = info.get("VERSION_ID", "")
            return f"{dist_id}{version_id}"
    except (OSError, AttributeError):
        pass
    # Fallback: try reading /etc/os-release directly
    try:
        with open("/etc/os-release") as f:
            info = {}
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    info[key] = value.strip('"')
            dist_id = info.get("ID", "linux").lower()
            version_id = info.get("VERSION_ID", "")
            return f"{dist_id}{version_id}"
    except (OSError, IOError):
        pass
    return "linux"


VALGRIND = os.getenv("VALGRIND", "0") == "1"
SANITIZER = os.getenv("SANITIZER", "") != ""
CODE_COVERAGE = os.getenv("CODE_COVERAGE", "0") == "1"

OS = _get_os()
ARCH = _get_arch()
OSNICK = _get_osnick()


def Env(
    moduleArgs=None,
    env="oss",
    useSlaves=False,
    enableDebugCommand=False,
    shardsCount=None,
):
    env = Environment(
        decodeResponses=True,
        moduleArgs=moduleArgs,
        env=env,
        useSlaves=useSlaves,
        enableDebugCommand=enableDebugCommand,
        shardsCount=shardsCount,
    )
    db = FalkorDB("localhost", env.port)

    if SANITIZER or VALGRIND:
        # patch env, turning every assert call into NOP
        # the arguments passed to the assert* are still evaluated
        # we've introduce this runtime patch to avoid any false asserts which
        # might be wrongly triggered running under memory sanitizer

        # The replacement function for self.env._assertion
        def sanitized_assertion_nop(self, checkStr, trueValue, depth=0, message=None):
            """
            Replaces the original _assertion function.

            1. It executes no logic internally.
            2. The 'trueValue' argument (the result of the assertion check)
               has already been calculated by the calling assert* function.
            3. No exception is raised, and no failure summary is recorded.
            """
            # Simply do nothing. The side effects (evaluation) have already happened.
            pass

        setattr(env, "_assertion", sanitized_assertion_nop.__get__(env))

    return (env, db)


def skip(cluster=False, macos=False):
    def decorate(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            env = x if isinstance(x, Env) else x.env
            if not (cluster or macos):
                env.skip()
            if cluster and env.isCluster():
                env.skip()
            if macos and OS == "macos":
                env.skip()
            return f(x, *args, **kwargs)

        return wrapper

    return decorate
