import time
import threading
import contextlib
from common import *

stop_event = threading.Event()
GRAPH_ID = "crash_handler"

# worker thread
def worker(db):
    while not stop_event.is_set():
        try:
            g = db.select_graph(GRAPH_ID)
            g.query("UNWIND range(0, 20000000) as x RETURN max(x), min(x)")
            time.sleep(0.1)
        except:
            return

def start_threads(db):
    stop_event.clear()

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(db,))
        t.start()
        threads.append(t)

    # give threads some time to start
    time.sleep(1)
    return threads

def stop_threads(db, threads):
    stop_event.set()

    # wait for all threads to finish
    for t in threads:
        t.join()

def validate_crash_report(env):
    # wait for the master process to exit
    env.envRunner.masterProcess.wait()

    # don't print the crash report to the console
    env.envRunner.masterProcess = None

    # verify we see a crash report
    logfilename = env.envRunner._getFileName("master", ".log")
    with open(f"{env.logDir}/{logfilename}") as logfile:
        log = logfile.read()

    assert "------ MODULES INFO OUTPUT ------" in log
    assert "# graph_executing commands" in log
    assert "=== REDIS BUG REPORT END" in log

class testMainThreadCrashHandler():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env(enableDebugCommand=True)
        cls.g = cls.db.select_graph(GRAPH_ID)

    def test_crash_main_thread(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # generate traffic
        workers = start_threads(self.db)

        with contextlib.suppress(Exception):
            # trigger a crash
            self.db.execute_command("DEBUG", "SEGFAULT")
            validate_crash_report(self.env)

        stop_threads(self.db, workers)

class testThreadOOM():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env(enableDebugCommand=True)
        cls.g = cls.db.select_graph(GRAPH_ID)

    def test_crash_thread_oom(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # generate traffic
        workers = start_threads(self.db)

        with contextlib.suppress(Exception):
            # trigger crash on one of the worker threads
            self.db.execute_command("GRAPH.DEBUG", "OOM")
            validate_crash_report(self.env)

        stop_threads(self.db, workers)

class testThreadAssert():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env(enableDebugCommand=True)
        cls.g = cls.db.select_graph(GRAPH_ID)

    def test_crash_thread_assert(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # generate traffic
        workers = start_threads(self.db)

        with contextlib.suppress(Exception):
            # trigger crash on one of the worker threads
            self.db.execute_command("GRAPH.DEBUG", "ASSERT")
            validate_crash_report(self.env)

        stop_threads(self.db, workers)

class testThreadSegFault():
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env(enableDebugCommand=True)
        cls.g = cls.db.select_graph(GRAPH_ID)

    def test_crash_thread_segfault(self):
        if SANITIZER:
            # Sanitizers are not compatible with the crash handler
            self.env.skip()
            return

        # generate traffic
        workers = start_threads(self.db)

        with contextlib.suppress(Exception):
            # trigger crash on one of the worker threads
            self.db.execute_command("GRAPH.DEBUG", "SEGFAULT")
            validate_crash_report(self.env)

        stop_threads(self.db, workers)

