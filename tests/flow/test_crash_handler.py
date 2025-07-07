from common import *
from index_utils import *
from constraint_utils import *

GRAPH_ID = "constraints"

class testCrashHandler():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)

    def test_crash_handler(self):
        try:
            self.db.execute_command("DEBUG", "SEGFAULT")
        except:
            pass
        logfilename = self.env.envRunner._getFileName("master", ".log")
        logfile = open(f"{self.env.logDir}/{logfilename}")
        log = logfile.read()
        self.env.assertGreater(len(log.splitlines()), 30)
