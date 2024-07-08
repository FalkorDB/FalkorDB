
import os
import sys
from subprocess import Popen, PIPE
import tempfile

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class ShError(Exception):
    def __init__(self, err, out=None, retval=None):
        super().__init__(err)
        self.out = out
        self.retval = retval

def sh(cmd, join=False, lines=False, fail=True):
    shell = isinstance(cmd, str)
    if shell:
        # Popen with shell=True defaults to /bin/sh so in order to use bash and
        # avoid quoting problems we write cmd into a temp file
        fd, cmd_file = tempfile.mkstemp(prefix='/tmp/sh')
        with open(cmd_file, 'w') as file:
            file.write(cmd)
        os.close(fd)
        cmd = "/usr/bin/env bash {cmd_file}".format(cmd_file=cmd_file)
    proc = Popen(cmd, shell=shell, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    if shell:
        os.unlink(cmd_file)
    out = out.decode('utf-8').strip()
    if lines is True:
        join = False
    if lines is True or join is not False:
        out = out.split("\n")
    if join is not False:
        s = join if type(join) is str else ' '
        out = s.join(out)
    if proc.returncode != 0 and fail is True:
        raise ShError(err.decode('utf-8'), out=out, retval=proc.returncode)
    return out
