import os
import inspect
import subprocess
import time
from common import *

GRAPH_ID = "config"
NUMBER_OF_CONFIGURATIONS = 22 # number of configurations available

class testConfig(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def tearDown(self):
        # Reset mutable runtime configs to defaults so changes in one test
        # don't leak into subsequent flow tests sharing the same Redis process.
        defaults = {
            "TIMEOUT": 0,
            "TIMEOUT_DEFAULT": 0,
            "TIMEOUT_MAX": 0,
            "RESULTSET_SIZE": -1,
            "MAX_QUEUED_QUERIES": 4294967295,
            "QUERY_MEM_CAPACITY": 0,
            "DELTA_MAX_PENDING_CHANGES": 10000,
            "VKEY_MAX_ENTITY_COUNT": 100000,
            "CMD_INFO": 1,
            "MAX_INFO_QUERIES": 1000,
            "EFFECTS_THRESHOLD": 300,
            "DELAY_INDEXING": 0,
            "JS_HEAP_SIZE": 256 * 1024 * 1024,
            "JS_STACK_SIZE": 1024 * 1024,
        }

        for name, value in defaults.items():
            try:
                self.db.config_set(name, value)
            except redis.exceptions.ResponseError:
                pass

    def test01_config_get(self):
        # Try reading 'QUERY_MEM_CAPACITY' from config
        config_name = "QUERY_MEM_CAPACITY"
        response = self.db.config_get(config_name)
        expected_response = 0 # capacity=QUERY_MEM_CAPACITY_UNLIMITED
        self.env.assertEqual(response, expected_response)

        # Try reading all configurations
        response = self.redis_con.execute_command("GRAPH.CONFIG GET *")

        # 16 configurations should be reported
        self.env.assertEquals(len(response), NUMBER_OF_CONFIGURATIONS)

        # validate default configuration values

        default_config = [
                ("TIMEOUT", 0),
                ("TIMEOUT_DEFAULT", 0),
                ("TIMEOUT_MAX",  0),
                ("CACHE_SIZE", 25),
                ("ASYNC_DELETE", [0,1]), # could be either 0 or 1 depending on load time config
                # OMP thread count can be 1 when OpenMP is unavailable.
                ("OMP_THREAD_COUNT", [1, os.cpu_count()]),
                ("THREAD_COUNT", os.cpu_count()),
                ("RESULTSET_SIZE", -1),
                ("VKEY_MAX_ENTITY_COUNT", 100000),
                ("MAX_QUEUED_QUERIES", 4294967295),
                ("QUERY_MEM_CAPACITY", 0),
                ("DELTA_MAX_PENDING_CHANGES", 10000),
                ("NODE_CREATION_BUFFER", 16384),
                ("CMD_INFO", 1),
                ("MAX_INFO_QUERIES", 1000),
                ("EFFECTS_THRESHOLD", 300),
                ("BOLT_PORT", 65535),
                ("DELAY_INDEXING", 0),
                ("IMPORT_FOLDER", "/var/lib/FalkorDB/import/"),
                ("TEMP_FOLDER", "/tmp"),
                ("JS_HEAP_SIZE", 256 * 1024 * 1024),
                ("JS_STACK_SIZE", 1024 * 1024)
        ]

        for i, config in enumerate(response):
            name  = config[0]
            value = config[1]

            # validate config name
            self.env.assertEquals(name, default_config[i][0])

            # validate config value
            if type(default_config[i][1]) is list:
                self.env.assertIn(value, default_config[i][1])
            else:
                self.env.assertEquals(value, default_config[i][1])

    def test02_config_get_invalid_name(self):
        # Ensure that getter fails on invalid parameters appropriately
        fake_config_name = "FAKE_CONFIG_NAME"

        try:
            self.db.config_get(fake_config_name)
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Unknown configuration field" in str(e))
            pass

    def test03_config_set(self):
        config_name = "RESULTSET_SIZE"
        config_value = 3

        # Set configuration
        response = self.db.config_set(config_name, config_value)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get(config_name)
        expected_response = config_value
        self.env.assertEqual(response, expected_response)

        config_name = "QUERY_MEM_CAPACITY"
        config_value = 1<<20 # 1MB

        # Set configuration
        response = self.db.config_set(config_name, config_value)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get(config_name)
        expected_response = config_value
        self.env.assertEqual(response, expected_response)

    def test04_config_set_multi(self):
        # Set multiple configuration values
        response = self.redis_con.execute_command("GRAPH.CONFIG SET RESULTSET_SIZE 3 QUERY_MEM_CAPACITY 100")
        self.env.assertEqual(response, "OK")

        # Make sure both values been updated
        names = ["RESULTSET_SIZE", "QUERY_MEM_CAPACITY"]
        values = [3, 100]
        for name, val in zip(names, values):
            response = self.db.config_get(name)
            expected_response = val
            self.env.assertEqual(response, expected_response)

    def test05_config_set_invalid_multi(self):
        # Get current configuration
        prev_conf = self.redis_con.execute_command("GRAPH.CONFIG GET *")

        try:
            # Set multiple configuration values, THREAD_COUNT is NOT
            # a runtime configuration, expecting this command to fail
            response = self.redis_con.execute_command("GRAPH.CONFIG SET QUERY_MEM_CAPACITY 150 THREAD_COUNT 40")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("This configuration parameter cannot be set at run-time" in str(e))

        try:
            # Set multiple configuration values, FAKE_CONFIG_NAME is NOT a valid
            # configuration, expecting this command to fail
            response = self.redis_con.execute_command("GRAPH.CONFIG SET QUERY_MEM_CAPACITY 150 FAKE_CONFIG_NAME 40")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Unknown configuration field" in str(e))

        try:
            # Set multiple configuration values, -1 is not a valid value for
            # MAX_QUEUED_QUERIES, expecting this command to fail
            response = self.redis_con.execute_command("GRAPH.CONFIG SET QUERY_MEM_CAPACITY 150 MAX_QUEUED_QUERIES -1")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Failed to set config value" in str(e))

        # make sure configuration wasn't modified
        current_conf = self.redis_con.execute_command("GRAPH.CONFIG GET *")
        self.env.assertEqual(prev_conf, current_conf)

    def test06_config_set_invalid_name(self):

        # Ensure that setter fails on unknown configuration field
        fake_config_name = "FAKE_CONFIG_NAME"

        try:
            self.db.config_set(fake_config_name, " 5")
            assert(False)
        except redis.exceptions.ResponseError as e:
            # Expecting an error.
            assert("Unknown configuration field" in str(e))
            pass

    def test07_config_invalid_subcommand(self):

        # Ensure failure on invalid sub-command, e.g. GRAPH.CONFIG DREP...
        config_name = "RESULTSET_SIZE"
        try:
            response = self.redis_con.execute_command("GRAPH.CONFIG DREP " + config_name + " 3")
            assert(False)
        except redis.exceptions.ResponseError as e:
            assert("Unknown subcommand for GRAPH.CONFIG" in str(e))
            pass

    def test08_config_reset_to_defaults(self):
        # Revert memory limit to default
        response = self.db.config_set("QUERY_MEM_CAPACITY", 0)
        self.env.assertEqual(response, "OK")

        # Change timeout value from default
        response = self.db.config_set("TIMEOUT", 5)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get("TIMEOUT")
        expected_response = 5
        self.env.assertEqual(response, expected_response)

        # Revert timeout to unlimited
        response = self.db.config_set("TIMEOUT", 0)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get("TIMEOUT")
        expected_response = 0
        self.env.assertEqual(response, expected_response)

        # Change timeout_default value from default
        response = self.db.config_set("TIMEOUT_DEFAULT", 5)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get("TIMEOUT_DEFAULT")
        expected_response = 5
        self.env.assertEqual(response, expected_response)

        # Revert timeout_default to unlimited
        response = self.db.config_set("TIMEOUT_DEFAULT", 0)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get("TIMEOUT_DEFAULT")
        expected_response = 0
        self.env.assertEqual(response, expected_response)

        # Change timeout_max value from default
        response = self.db.config_set("TIMEOUT_MAX", 5)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get("TIMEOUT_MAX")
        expected_response = 5
        self.env.assertEqual(response, expected_response)

        # Revert timeout_max to unlimited
        response = self.db.config_set("TIMEOUT_MAX", 0)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get("TIMEOUT_MAX")
        expected_response = 0
        self.env.assertEqual(response, expected_response)

        # Change resultset_size from default
        response = self.db.config_set("RESULTSET_SIZE", 2)
        self.env.assertEqual(response, "OK")

        # Validate modified resultset_size
        result = self.graph.query("UNWIND range(1, 10) AS v RETURN v")
        self.env.assertEqual(len(result.result_set), 2)

        # Revert resultset_size to unlimited with a negative argument
        response = self.db.config_set("RESULTSET_SIZE", -100)
        self.env.assertEqual(response, "OK")

        # Make sure resultset_size has been updated to unlimited.
        response = self.db.config_get("RESULTSET_SIZE")
        expected_response = -1
        self.env.assertEqual(response, expected_response)

        response = self.db.config_get("NODE_CREATION_BUFFER")
        expected_response = 16384
        self.env.assertEqual(response, expected_response)

        response = self.db.config_get("DELAY_INDEXING")
        expected_response = 0
        self.env.assertEqual(response, expected_response)

        response = self.db.config_set("JS_HEAP_SIZE", 256 * 1024 * 1024)
        self.env.assertEqual(response, "OK")

        response = self.db.config_get("JS_HEAP_SIZE")
        expected_response = 256 * 1024 * 1024
        self.env.assertEqual(response, expected_response)

        response = self.db.config_set("JS_STACK_SIZE", 1024 * 1024)
        self.env.assertEqual(response, "OK")

        response = self.db.config_get("JS_STACK_SIZE")
        expected_response = 1024 * 1024
        self.env.assertEqual(response, expected_response)

    def test09_set_invalid_values(self):
        # The run-time configurations supported by RedisGraph are:
        # MAX_QUEUED_QUERIES
        # TIMEOUT
        # QUERY_MEM_CAPACITY
        # DELTA_MAX_PENDING_CHANGES
        # RESULTSET_SIZE

        # Validate that attempting to set these configurations to
        # invalid values fails
        try:
            # MAX_QUEUED_QUERIES must be a positive value
            self.db.config_set("MAX_QUEUED_QUERIES", 0)
            assert(False)
        except redis.exceptions.ResponseError as e:
            assert("Failed to set config value MAX_QUEUED_QUERIES to 0" in str(e))
            pass

        # TIMEOUT, QUERY_MEM_CAPACITY, and DELTA_MAX_PENDING_CHANGES must be
        # non-negative values, 0 resets to default
        for config in ["TIMEOUT", "QUERY_MEM_CAPACITY", "DELTA_MAX_PENDING_CHANGES"]:
            try:
                self.db.config_set(f"{config}", -1)
                assert(False)
            except redis.exceptions.ResponseError as e:
                assert("Failed to set config value %s to -1" % config in str(e))
                pass

        # No configuration can be set to a string
        for config in ["MAX_QUEUED_QUERIES", "TIMEOUT", "QUERY_MEM_CAPACITY",
                       "DELTA_MAX_PENDING_CHANGES", "RESULTSET_SIZE"]:
            try:
                self.db.config_set(config, "invalid")
                assert(False)
            except redis.exceptions.ResponseError as e:
                assert(("Failed to set config value %s to invalid" % config) in str(e))

    def test10_set_get_vkey_max_entity_count(self):
        config_name = "VKEY_MAX_ENTITY_COUNT"
        config_value = 100

        # Set configuration
        response = self.db.config_set(config_name, config_value)
        self.env.assertEqual(response, "OK")

        # Make sure config been updated.
        response = self.db.config_get(config_name)
        expected_response = config_value
        self.env.assertEqual(response, expected_response)

    def test11_set_get_node_creation_buffer(self):
        # flush and stop is needed for memcheck for clean shutdown
        self.graph.delete()
        self.env.stop()

        self.env, self.db = Env(moduleArgs='NODE_CREATION_BUFFER 0')
        self.redis_con = self.env.getConnection()

        # values less than 128 (such as 0, which this module was loaded with)
        # will be increased to 128
        creation_buffer_size = self.db.config_get("NODE_CREATION_BUFFER")
        expected_response =  128
        self.env.assertEqual(creation_buffer_size, expected_response)

        # restart the server with a buffer argument of 600
        self.env, self.db = Env(moduleArgs='NODE_CREATION_BUFFER 600')
        self.redis_con = self.env.getConnection()

        # the node creation buffer should be 1024, the next-greatest power of 2 of 600
        creation_buffer_size = self.db.config_get("NODE_CREATION_BUFFER")
        expected_response = 1024
        self.env.assertEqual(creation_buffer_size, expected_response)

    def test12_config_commands_available_via_redis_config(self):
        timeout_key = "graph.TIMEOUT"

        # ensure CONFIG GET exposes module configs
        cfg = self.redis_con.config_get("graph.*")
        self.env.assertIn(timeout_key, cfg)

        # set via CONFIG SET and validate through GRAPH.CONFIG path
        self.redis_con.config_set(timeout_key, 2)
        self.env.assertEqual(self.db.config_get("TIMEOUT"), 2)

        cfg = self.redis_con.config_get(timeout_key)
        self.env.assertEqual(int(cfg[timeout_key]), 2)

        # reset
        self.redis_con.config_set(timeout_key, 0)

import stat
import shutil
import tempfile

class testConfigTempFolder:
    def __init__(self):
        self.env, self.db = Env()
        if SANITIZER or VALGRIND:
            self.env.skip()

    def teardown_method(self):
        if hasattr(self, 'conn'):
            self.conn.shutdown()

    def set_temp_folder(self, path):
        module_args = f"TEMP_FOLDER {path}"
        self.env, self.db = Env(moduleArgs=module_args, enableDebugCommand=True)

        self.conn = self.env.getConnection()

    def test_01_temp_folder_is_file(self):
        # try setting TEMP_FOLDER to a file
        # expecting config update to fail
        fd, file_path = tempfile.mkstemp()
        os.close(fd)

        # try updating TEMP_FOLDER
        try:
            self.set_temp_folder(file_path)
            # setting TEMP_FOLDER to a file should have failed
            self.env.assertFalse(True)
        except Exception:
            pass

    def test_02_temp_folder_not_exist(self):
        # try setting TEMP_FOLDER to a non existing folder
        # expecting config update to fail

        # make sure path doesn't exists
        non_existent = "/tmp/falkordb_nonexistent_dir"
        if os.path.exists(non_existent):
            shutil.rmtree(non_existent)

        # try updating TEMP_FOLDER
        try:
            self.set_temp_folder(non_existent)
            self.env.assertFalse(True)
        except Exception:
            pass

    def test_03_temp_folder_no_permission(self):
        # try setting TEMP_FOLDER to a folder which we can't write to
        # expecting config update to fail, as write access is mandatory

        # create a temp folder with no write access
        no_perm_dir = tempfile.mkdtemp()
        os.chmod(no_perm_dir, stat.S_IREAD)

        # check if directory is truly unwritable
        if os.access(no_perm_dir, os.W_OK):
            env, _ = Env(enableDebugCommand=True)
            env.skip()

        # try updating TEMP_FOLDER
        try:
            self.set_temp_folder(no_perm_dir)
            # setting TEMP_FOLDER to a non writeable folder should have failed
            self.env.assertFalse(True)
        except Exception:
            pass
        finally:
            # clean up
            os.chmod(no_perm_dir, stat.S_IWUSR | stat.S_IREAD | stat.S_IXUSR)
            shutil.rmtree(no_perm_dir)

    def test_04_temp_folder_exists_success(self):
        # try setting TEMP_FOLDER to a valid folder
        # expecting config update to succeed

        valid_dir = tempfile.mkdtemp()

        try:
            self.set_temp_folder(valid_dir)
            self.env.assertEqual(self.db.config_get("TEMP_FOLDER"), valid_dir)
        finally:
            # clean up
            shutil.rmtree(valid_dir)

class testConfigRewritePersist:
    def _reset_runtime_configs(self):
        if not hasattr(self, "db"):
            return

        defaults = {
            "TIMEOUT": 0,
            "TIMEOUT_DEFAULT": 0,
            "TIMEOUT_MAX": 0,
            "RESULTSET_SIZE": -1,
            "CMD_INFO": 1,
        }

        for name, value in defaults.items():
            try:
                self.db.config_set(name, value)
            except Exception:
                pass

    def teardown_method(self):
        self._reset_runtime_configs()

        if hasattr(self, "env"):
            self.env.stop()
        if hasattr(self, "redis_proc") and self.redis_proc:
            try:
                self.redis_con.shutdown()
            except Exception:
                pass
            self.redis_proc.terminate()
            try:
                self.redis_proc.wait(timeout=5)
            except Exception:
                self.redis_proc.kill()
        if hasattr(self, "cfg_path") and os.path.exists(self.cfg_path):
            os.remove(self.cfg_path)

    def test_config_rewrite_roundtrip(self):
        # configure via redis.conf only, no module args
        cfg_lines = [
            "graph.TIMEOUT 7",
            "graph.RESULTSET_SIZE 4",
            "graph.CMD_INFO no",
        ]

        fd, self.cfg_path = tempfile.mkstemp(prefix="falkordb-config-", suffix=".conf")
        with os.fdopen(fd, "w") as cfg:
            cfg.write("\n".join(cfg_lines))

        params = inspect.signature(Env).parameters
        env_kwargs = {}
        if "redisConfigFile" in params:
            env_kwargs["redisConfigFile"] = self.cfg_path
        elif "redisConfigPath" in params:
            env_kwargs["redisConfigPath"] = self.cfg_path
        elif "redisConfig" in params:
            env_kwargs["redisConfig"] = self.cfg_path

        if env_kwargs:
            self.env, self.db = Env(**env_kwargs)
            self.redis_con = self.env.getConnection()
            self.redis_proc = None
        else:
            # RLTest version does not support passing a config file; launch Redis
            # manually using the same binary/module args RLTest would use.
            self.env, _ = Env()
            runner = self.env.envRunner
            port = runner.port

            # pull loadmodule directives from the RLTest command and move them into the config
            loadmodule_lines = []
            filtered_args = []
            args = runner.masterCmdArgs[1:]  # skip redis-server binary
            i = 0
            while i < len(args):
                if args[i] == "--loadmodule" and i + 1 < len(args):
                    j = i + 2
                    while j < len(args) and not args[j].startswith("--"):
                        j += 1
                    load_args = args[i + 1 : j]
                    loadmodule_lines.append("loadmodule " + " ".join(load_args))
                    i = j
                else:
                    filtered_args.append(args[i])
                    i += 1

            if loadmodule_lines:
                with open(self.cfg_path, "r+") as cfg:
                    existing = cfg.read()
                    cfg.seek(0)
                    cfg.write("\n".join(loadmodule_lines) + "\n" + existing)
                    cfg.truncate()

            # stop the RLTest-managed instance so we can reuse the port/args
            self.env.stop()

            cmd = [runner.redisBinaryPath, self.cfg_path] + filtered_args
            self.redis_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )

            self.redis_con = redis.StrictRedis(
                "localhost", port, decode_responses=True
            )
            for _ in range(100):
                try:
                    if self.redis_con.ping():
                        break
                except Exception:
                    time.sleep(0.05)
            self.db = FalkorDB("localhost", port)

        # values should load from config file
        self.env.assertEqual(self.db.config_get("TIMEOUT"), 7)
        self.env.assertEqual(self.db.config_get("RESULTSET_SIZE"), 4)
        self.env.assertEqual(self.db.config_get("CMD_INFO"), 0)

        # rewrite and ensure persisted in file
        rewrite_result = self.redis_con.config_rewrite()
        self.env.assertIn(rewrite_result, [True, "OK"])
        with open(self.cfg_path, "r") as cfg:
            contents = cfg.read().lower()

        for line in cfg_lines:
            key, val = line.split()
            self.env.assertIn(f"{key.lower()} {val.lower()}", contents)

        # still reflected in the running config
        self.env.assertEqual(self.db.config_get("TIMEOUT"), 7)
        self.env.assertEqual(self.db.config_get("RESULTSET_SIZE"), 4)
        self.env.assertEqual(self.db.config_get("CMD_INFO"), 0)
