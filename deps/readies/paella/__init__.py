
import sys

from .error import *
from .classes import *
from .debug import *
from .utils import *
from .files import *
from .func import *
from .text import *
from .log import *
from .platform import *
from .setup import *
from .contrib.version import Version

#----------------------------------------------------------------------------------------------

class global_injector:
    def __init__(self):
        try:
            # Python 2
            self.__dict__['builtin'] = sys.modules['__builtin__'].__dict__
        except KeyError:
            # Python 3
            self.__dict__['builtin'] = sys.modules['builtins'].__dict__
    def __setattr__(self,name,value):
        self.builtin[name] = value

Global = global_injector()

#----------------------------------------------------------------------------------------------

Global.BB = bb
Global.eprint = eprint
Global.fatal = fatal
Global.cwd = cwd
Global.sh = sh
Global.ctor = ctor
Global.noctor = noctor
Global.foldl = foldl
Global.foldr = foldr
Global.ENV = Env()
