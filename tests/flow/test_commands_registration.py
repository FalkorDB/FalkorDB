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
    @classmethod
    def setup_class(cls):
        cls.env, cls.db = Env()
        cls.conn = cls.env.getConnection()

    def test_deny_script(self):
        """Make sure none of the graph commands can run within LUA"""

        # register the script and get a Script object
        script = self.conn.register_script(LUA_SCRIPT)

        # collect all graph.* commands
        graph_commands = self.conn.execute_command("COMMAND list filterby module graph")
        assert len(graph_commands) > 0

        for cmd in graph_commands:
            try:
                result = script(args=[cmd])
                assert not (True and "Expecting script invocation to fail")
            except ResponseError as e:
                assert "This Redis command is not allowed from script" in str(e)

    def test_command_metadata_flags(self):
        def _command_flags(command):
            info = self.conn.execute_command("COMMAND", "INFO", command)
            if isinstance(info, dict):
                command_info = next(iter(info.values()))
                return set(command_info["flags"])
            return set(info[0][2])

        assert "denyoom" in _command_flags("graph.query")
        assert "denyoom" in _command_flags("graph.profile")
        assert "denyoom" in _command_flags("graph.constraint")
        assert "denyoom" in _command_flags("graph.copy")

        assert "allow_busy" in _command_flags("graph.config")
        assert "allow_busy" in _command_flags("graph.slowlog")
        assert "allow_busy" not in _command_flags("graph.query")
