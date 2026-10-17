from common import *

GRAPH_ID = "cmd_reg"

# Define the Lua script
LUA_SCRIPT = """
-- Redis Lua script for executing graph command
-- ARGV[1]: graph command name

local graph_cmd = ARGV[1]

-- Execute the graph command
local result
    result = redis.call(graph_cmd)

-- Return the result
return result
"""

class testCmdReg(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test_deny_script(self):
        """Make sure none of the graph commands can run within LUA"""

        # register the script and get a Script object
        script = self.conn.register_script(LUA_SCRIPT)

        # collect all graph.* commands
        graph_commands = self.conn.execute_command("COMMAND list filterby module graph")
        self.env.assertGreater(len(graph_commands), 0)

        for cmd in graph_commands:
            try:
                result = script(args=[cmd])
                self.env.assertFalse(True and "Expecting script invocation to fail")
            except ResponseError as e:
                self.env.assertContains("This Redis command is not allowed from script", str(e))

    def test_command_metadata_flags(self):
        def _command_flags(command):
            info = self.conn.execute_command("COMMAND", "INFO", command)
            if isinstance(info, dict):
                command_info = next(iter(info.values()))
                return set(command_info["flags"])
            return set(info[0][2])

        self.env.assertContains("denyoom", _command_flags("graph.query"))
        self.env.assertContains("denyoom", _command_flags("graph.profile"))
        self.env.assertContains("denyoom", _command_flags("graph.constraint"))
        self.env.assertContains("denyoom", _command_flags("graph.copy"))

        self.env.assertContains("allow_busy", _command_flags("graph.config"))
        self.env.assertContains("allow_busy", _command_flags("graph.slowlog"))
        self.env.assertNotContains("allow_busy", _command_flags("graph.query"))

    def test_info_has_no_redisearch_sections(self):
        """RediSearch registers an INFO callback on the graph module when it
        initialises; ours must replace it, as C's does, or `INFO` reports
        RediSearch's version as `graph_version` (#2915)."""

        # the module itself is still loaded
        names = [m[b"name"] if b"name" in m else m.get("name")
                 for m in self.conn.module_list()]
        self.env.assertTrue(any(n in ("graph", b"graph") for n in names))

        # and adds no sections of its own outside a crash report
        leaked = [k for k in self.conn.info("everything") if k.startswith("graph_")]
        self.env.assertEqual(leaked, [])
        self.env.assertEqual(self.conn.info("graph"), {})
