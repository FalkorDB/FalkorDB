import os
from common import *

GRAPH_ID = "config"
NUMBER_OF_CONFIGURATIONS = 20 # number of configurations available

class testConfig(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

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
                ("OMP_THREAD_COUNT", os.cpu_count()),
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
                ("DEDUPLICATE_STRINGS", 0),
                ("USE_DISK_STORAGE", 0),
                ("VALUES_SPILL_THRESHOLD", 64)
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

        response = self.db.config_get("USE_DISK_STORAGE")
        expected_response = 0
        self.env.assertEqual(response, expected_response)

        response = self.db.config_get("VALUES_SPILL_THRESHOLD")
        expected_response = 64
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

