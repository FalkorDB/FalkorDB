from common import Env

GRAPH_ID = "crash_report"

class testCrashHandler():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)

    def test_crash_handler(self):
        try:
            # trigger a crash
            self.db.execute_command("DEBUG", "SEGFAULT")
        except:
            pass

        # verify we see a crash report
        logfilename = self.env.envRunner._getFileName("master", ".log")
        with open(f"{self.env.logDir}/{logfilename}") as logfile:
            log = logfile.read()

        self.env.assertGreater(len(log.splitlines()), 30)

