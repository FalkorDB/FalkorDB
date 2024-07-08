
from __future__ import absolute_import
from .log import caller_info

#----------------------------------------------------------------------------------------------

class Error(RuntimeError):
    pass

#----------------------------------------------------------------------------------------------

def fatal(text):
    eprint("%s: %s" % (caller_info(1), text))
    exit(1)
