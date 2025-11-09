import os
import sys
from functools import wraps

from RLTest import Env as Environment, Defaults

import redis
from redis import ResponseError
from falkordb import FalkorDB, Graph, Node, Edge, Path, ExecutionPlan

from base import FlowTestsBase

Defaults.decode_responses = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/readies"))
import paella

VALGRIND      = os.getenv('VALGRIND', '0')      == '1'
SANITIZER     = os.getenv('SANITIZER', '')      != ''
CODE_COVERAGE = os.getenv('CODE_COVERAGE', '0') == '1'

OS     = paella.Platform().os
ARCH   = paella.Platform().arch
OSNICK = paella.Platform().osnick

def Env(moduleArgs=None, env='oss', useSlaves=False, enableDebugCommand=False):
    env = Environment(decodeResponses=True, moduleArgs=moduleArgs, env=env,
                      useSlaves=useSlaves, enableDebugCommand=enableDebugCommand)
    db  = FalkorDB("localhost", env.port)
    return (env, db)

def skip(cluster=False, macos=False):
    def decorate(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            env = x if isinstance(x, Env) else x.env
            if not(cluster or macos):
                env.skip()
            if cluster and env.isCluster():
                env.skip()
            if macos and OS == 'macos':
                env.skip()
            return f(x, *args, **kwargs)
        return wrapper
    return decorate
