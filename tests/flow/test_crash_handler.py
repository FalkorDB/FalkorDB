import contextlib
from common import *

class testCrashHandler():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)

    def test_crash_handler(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        with contextlib.suppress(Exception):
            # trigger a crash
            self.db.execute_command("DEBUG", "SEGFAULT")

        # wait for the master process to exit
        self.env.envRunner.masterProcess.wait()
        # don't print the crash report to the console
        self.env.envRunner.masterProcess = None

        # verify we see a crash report
        logfilename = self.env.envRunner._getFileName("master", ".log")
        with open(f"{self.env.logDir}/{logfilename}") as logfile:
            log = logfile.read()

        self.env.assertContains("------ MODULES INFO OUTPUT ------", log)
        self.env.assertContains("# graph_executing commands", log)
        self.env.assertContains("=== REDIS BUG REPORT END", log)

