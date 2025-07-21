import contextlib
from common import *

GRAPH_ID = "crash_handler"

def validate_crash_report(env):
    # wait for the master process to exit
    env.envRunner.masterProcess.wait()

    # don't print the crash report to the console
    env.envRunner.masterProcess = None

    # verify we see a crash report
    logfilename = env.envRunner._getFileName("master", ".log")
    with open(f"{env.logDir}/{logfilename}") as logfile:
        log = logfile.read()

    env.assertContains("------ MODULES INFO OUTPUT ------", log)
    env.assertContains("# graph_executing commands", log)
    env.assertContains("=== REDIS BUG REPORT END", log)

class testMainThreadCrashHandler():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.g = self.db.select_graph(GRAPH_ID)

    def test_crash_main_thread(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        with contextlib.suppress(Exception):
            # trigger a crash
            self.db.execute_command("DEBUG", "SEGFAULT")
            validate_crash_report(self.env)

class testThreadOOM():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.g = self.db.select_graph(GRAPH_ID)

    def test_crash_thread_oom(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        with contextlib.suppress(Exception):
            # trigger crash on one of the worker threads
            self.db.execute_command("GRAPH.DEBUG", "SEGFAULT")
            validate_crash_report(self.env)

class testThreadAssert():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.g = self.db.select_graph(GRAPH_ID)

    def test_crash_thread_assert(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        with contextlib.suppress(Exception):
            # trigger crash on one of the worker threads
            self.db.execute_command("GRAPH.DEBUG", "ASSERT")
            validate_crash_report(self.env)

class testThreadSegFault():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.g = self.db.select_graph(GRAPH_ID)

    def test_crash_thread_segfault(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        with contextlib.suppress(Exception):
            # trigger crash on one of the worker threads
            self.db.execute_command("GRAPH.DEBUG", "SEGFAULT")
            validate_crash_report(self.env)

