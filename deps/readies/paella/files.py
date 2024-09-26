
from contextlib import contextmanager
import errno
import os
import os.path
import shutil
import sys
import tempfile
try:
    from urllib2 import urlopen
except:
    from urllib.request import urlopen

#----------------------------------------------------------------------------------------------

def fread(fname, mode='r'):
    with open(fname, mode) as file:
        return file.read()

#----------------------------------------------------------------------------------------------

def fwrite(fname, text, mode='w', encode=True):
    with open(fname, mode) as file:
        return file.write(text)

#----------------------------------------------------------------------------------------------

def flines(fname, mode = 'r'):
    return [line.rstrip() for line in open(fname)]

#----------------------------------------------------------------------------------------------

def tempfilepath(prefix=None, suffix=None):
    if sys.version_info < (3, 0):
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path

#----------------------------------------------------------------------------------------------

def wget(url, dest="", tempdir=False):
    if dest == "":
        dest = os.path.basename(url)
        if dest == "":
            dest = tempfilepath()
        elif tempdir:
            dir = tempfile.mkdtemp()
            dest = os.path.join(dir, dest)
    else:
        if tempdir:
            dir = tempfile.mkdtemp()
            dest = os.path.join(dir, dest)
    ufile = urlopen(url)
    data = ufile.read()
    with open(dest, "wb") as file:
        file.write(data)
    return os.path.abspath(dest)

#----------------------------------------------------------------------------------------------

@contextmanager
def cwd(path):
    d0 = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(d0)

#----------------------------------------------------------------------------------------------

def mkdir_p(dir):
    if dir == '':
        return
    try:
        return os.makedirs(dir, exist_ok=True)
    except TypeError:
        pass
    try:
        return os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST or os.path.isfile(dir):
            raise

#----------------------------------------------------------------------------------------------

def rm_rf(path, careful=True):
    if careful and os.path.normpath(path) in [".", "..", "/", "//", ""]:
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

#----------------------------------------------------------------------------------------------

def relpath(dir, rel):
    return os.path.abspath(os.path.join(dir, rel))
