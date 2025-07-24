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

