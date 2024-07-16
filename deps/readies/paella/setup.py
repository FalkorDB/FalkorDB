
import os
import sys
import re
import shutil
import subprocess
import tempfile
import textwrap
from .platform import OnPlatform, Platform
from .error import *
import paella

GIT_LFS_VER = '2.12.1'

#----------------------------------------------------------------------------------------------

class OutputMode:
    def __init__(self, x):
        lx = str(x).lower()
        if x == True or x == 1 or lx == "yes" or lx == "true":
            self.mode = "True"
        elif x == False or x == 0 or lx == "no" or lx == "false":
            self.mode = "False"
        elif lx == "on_error":
            self.mode = "on_error"
        else:
            raise Error("Wrong output mode: %s" % x)

    def __eq__(self, x):
        return self.mode == OutputMode(x).mode

    def __ne__(self, x):
        return not self.__eq__(x)

    def __bool__(self):
        return self.mode == "True"

    def on_error(self):
        return self.mode == "on_error"

#----------------------------------------------------------------------------------------------

class Runner:
    def __init__(self, nop=False, output="on_error"):
        self.nop = nop
        self.is_root = os.geteuid() == 0
        self.has_sudo = sh('command -v sudo', fail=False) != ''
        self.output = OutputMode(output)

    # sudo: True/False/"file"
    def run(self, cmd, at=None, output=None, nop=None, _try=False, sudo=False, echo=True):
        # We're running cmd(s) with a login shell ("bash -l") in order to run profile.d
        # scripts (installation commands may add such scripts and subsequent installation
        # commands may rely on them).
        # Howerver, "bash -l" will wreck PATH of active virtualenvs, thus python scripts will
        # fail. So if we're in one (i.e. VIRTUAL_ENV is not empty) PATH sould be restored
        # by re-invoking the activation script.
        if (self.is_root or not self.has_sudo) and sudo is not False:
            sudo = False
        if output is None:
            output = self.output
        else:
            output = OutputMode(output)
        venv = ENV['VIRTUAL_ENV']
        cmd_file = None
        if cmd.find('\n') > -1:
            cmds1 = str.lstrip(textwrap.dedent(cmd))
            cmds = list(filter(lambda s: str.lstrip(s) != '', cmds1.split("\n")))
            if venv != '':
                cmds = [". {VENV}/bin/activate".format(VENV=venv)] + cmds
            cmd = "; ".join(cmds)
            cmd_for_log = cmd
            if sudo is not False:
                cmd_file = paella.tempfilepath()
                paella.fwrite(cmd_file, cmd)
                cmd = "bash -l {}".format(cmd_file)
                cmd_for_log = "sudo { %s }" % cmd_for_log
        else:
            if venv != '':
                cmd = "{{ . {VENV}/bin/activate; {CMD}; }}".format(VENV=venv, CMD=cmd)
            cmd_for_log = cmd
        if sudo is not False:
            if sudo == "file":
                cmd_file = paella.tempfilepath()
                paella.fwrite(cmd_file, cmd)
                cmd = "sudo bash -l {}".format(cmd_file)
                cmd_for_log = "sudo { %s }" % cmd_for_log
            else:
                cmd = "sudo bash -l -c '{}'".format(cmd)
        if echo:
            print(cmd)
        if cmd_file is not None:
            print("# {}".format(cmd_for_log))
        sys.stdout.flush()
        if nop is None:
            nop = self.nop
        if nop:
            return
        if output != True:
            fd, temppath = tempfile.mkstemp()
            os.close(fd)
            cmd = "{{ {CMD}; }} >{LOG} 2>&1".format(CMD=cmd, LOG=temppath)
        if at is None:
            rc = subprocess.call(["bash", "-l", "-e", "-c", cmd])
        else:
            with cwd(at):
                # rc = os.system(cmd)
                rc = subprocess.call(["bash", "-l", "-e", "-c", cmd])
        if rc > 0:
            if output != True:
                if output.on_error():
                    os.system("cat {}".format(temppath))
                eprint("command failed: " + cmd_for_log)
                sys.stderr.flush()
        if output != True:
            os.remove(temppath)
        if cmd_file is not None:
            os.remove(cmd_file)
        if rc > 0 and not _try:
            sys.exit(1)
        return rc

    def has_command(self, cmd):
        return Runner.has_command(cmd)

    @staticmethod
    def has_command(cmd):
        return os.system("command -v " + cmd + " > /dev/null") == 0

#----------------------------------------------------------------------------------------------

class PackageManager(object):
    def __init__(self, runner):
        self.runner = runner

    @staticmethod
    def detect(platform, runner):
        if platform.os == 'linux':
            if platform.is_debian_compat():
                return Apt(runner)
            elif platform.is_redhat_compat():
                return Yum(runner) if platform.redhat_compat_version() == 7 else Dnf(runner)
            elif platform.dist == 'mariner':
                return TDnf(runner)
            elif platform.dist == 'fedora':
               return Dnf(runner)
            elif platform.dist == 'suse':
                return Zypper(runner)
            elif platform.dist == 'arch':
                return Pacman(runner)
            elif platform.dist == 'alpine':
                return Alpine(runner)
            else:
                raise Error("Cannot determine package manager for distibution %s" % platform.dist)
        elif platform.os == 'macos':
            return Brew(runner)
        elif platform.os == 'freebsd':
            return Pkg(runner)
        else:
            raise Error("Cannot determine package manager for OS %s" % platform.os)

    def run(self, cmd, at=None, output="on_error", nop=None, _try=False, sudo=False, echo=True):
        return self.runner.run(cmd, at=at, output=output, nop=nop, _try=_try, sudo=sudo)

    def has_command(self, cmd):
        return self.runner.has_command(cmd)

    def install(self, packs, group=False, output="on_error", _try=False):
        return False

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return False

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        return False

    def update(self, output="on_error"):
        return False

#----------------------------------------------------------------------------------------------

class Yum(PackageManager):
    def __init__(self, runner):
        super(Yum, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False):
        if not group:
            return self.run("yum install -q -y " + packs, output=output, _try=_try, sudo=True)
        else:
            return self.run("yum groupinstall -y " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        if not group:
            return self.run("yum remove -q -y " + packs, output=output, _try=_try, sudo=True)
        else:
            return self.run("yum group remove -y " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        if not self.has_command("yum-config-manager"):
            return self.install("yum-utils")
        return self.run("yum-config-manager -y --add-repo {}".format(repourl), output=output, _try=_try, sudo=True)

#----------------------------------------------------------------------------------------------

class Dnf(PackageManager):
    def __init__(self, runner):
        super(Dnf, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False):
        if not group:
            return self.run("dnf install -q -y " + packs, output=output, _try=_try, sudo=True)
        else:
            return self.run("dnf groupinstall -y " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        if not group:
            return self.run("dnf remove -q -y " + packs, output=output, _try=_try, sudo=True)
        else:
            return self.run("dnf group remove -y " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        if self.run("dnf config-manager 2>/dev/null", output=output, _try=True):
            return self.install("dnf-plugins-core", _try=_try)
        return self.run("dnf config-manager -y --add-repo {}".format(repourl), output=output, _try=_try, sudo=True)

#----------------------------------------------------------------------------------------------

class TDnf(PackageManager):
    def __init__(self, runner):
        super(TDnf, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False):
        if not group:
            return self.run("tdnf install -q -y " + packs, output=output, _try=_try, sudo=True)
        else:
            return self.run("tdnf groupinstall -y " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        if not group:
            return self.run("tdnf remove -q -y " + packs, output=output, _try=_try, sudo=True)
        else:
            return self.run("tdnf group remove -y " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        if self.run("tdnf config-manager 2>/dev/null", output=output, _try=True):
            return self.install("tdnf-plugins-core", _try=_try)
        return self.run("tdnf config-manager -y --add-repo {}".format(repourl), output=output, _try=_try, sudo=True)

#----------------------------------------------------------------------------------------------

class Apt(PackageManager):
    def __init__(self, runner):
        super(Apt, self).__init__(runner)

        # prevents apt-get from interactively prompting
        os.environ["DEBIAN_FRONTEND"] = 'noninteractive'

    def install(self, packs, group=False, output="on_error", _try=False):
        return self.run("apt-get -qq install --fix-missing -y " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return self.run("apt-get -qq remove -y " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repo_url, repo="", output="on_error", _try=False):
        if not self.has_command("add-apt-repository"):
            self.install("software-properties-common")
        rc = self.run("add-apt-repository -y {URL}".format(URL=repo_url), output=output, _try=_try, sudo=True)
        self.run("apt-get -qq update", output=output, _try=_try, sudo=True)
        return rc

    def update(self, output="on_error"):
        return self.run("apt-get -qq update -y", output=output, sudo=True)

#----------------------------------------------------------------------------------------------

class Zypper(PackageManager):
    def __init__(self, runner):
        super(Zypper, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False):
        return self.run("zypper --non-interactive install " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return self.run("zypper --non-interactive remove " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repo_url, repo="", output="on_error", _try=False):
        return self.run("zypprt addrepo {URL} {NAME}".format(URL=repo_url, NAME=repo), output=output, _try=_try, sudo=True)

#----------------------------------------------------------------------------------------------

class Pacman(PackageManager):
    def __init__(self, runner):
        super(Pacman, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False, aur=False):
        if aur is False:
            return self.run("pacman --noconfirm -S " + packs, output=output, _try=_try, sudo=True)
        else:
            if os.path.isfile("/usr/bin/yay"):
                aurbin = "yay"
            if os.path.isfile("/usr/bin/trizen"):
                aurbin = "trizen"
            else:
                raise FileNotFoundError("Failed to find yay or trizen, for aur package installation.")
            return self.run("{} --noconfirm -S {}".format(aurbin, packs), output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return self.run("pacman --noconfirm -R " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        return False

#----------------------------------------------------------------------------------------------

class Brew(PackageManager):
    def __init__(self, runner):
        super(Brew, self).__init__(runner)

        # prevents brew from performing auto updates
        os.environ["HOMEBREW_NO_AUTO_UPDATE"] = "1"

        if os.getuid() == 0 and os.getenv("BREW_AS_ROOT") != "1":
            eprint("Cannot run as root. Set BREW_AS_ROOT=1 to override.")
            sys.exit(1)
        if sh('xcode-select -p') == '':
            eprint("Xcode tools are not installed. Please run xcode-select --install.")
            sys.exit(1)
        if sys.version_info < (3, 0):
            if 'VIRTUAL_ENV' not in os.environ:
                # required because osx pip installed are done with --user
                os.environ["PATH"] = os.environ["PATH"] + ':' + os.environ["HOME"] + '/Library/Python/2.7/bin'

    def install(self, packs, group=False, output="on_error", _try=False):
        # brew will fail if package is already installed
        rc = True
        for pack in packs.split():
            rc = self.run("brew list {PACK} &>/dev/null || brew install {PACK}".format(PACK=pack),
                     output=output, _try=_try, sudo=False) and rc
        return rc

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        rc = True
        for pack in packs.split():
            rc = self.run("brew remove {PACK}".format(PACK=pack), output=output, _try=_try,
                          sudo=False) and rc
        return rc

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        return False

    def update(self, output="on_error"):
        if os.environ.get('BREW_UPDATE') != '1':
            return True
        return self.run("brew update || true", output=output)

#----------------------------------------------------------------------------------------------

class Pkg(PackageManager):
    def __init__(self, runner):
        super(Pkg, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False):
        return self.run("pkg install -q -y " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return self.run("pkg delete -q -y " + packs, output=output, _try=_try, sudo=True)

    def add_repo(self, repourl, repo="", output="on_error", _try=False):
        return False

#----------------------------------------------------------------------------------------------

class Alpine(PackageManager):
    def __init__(self, runner):
        super(Alpine, self).__init__(runner)

    def install(self, packs, group=False, output="on_error", _try=False):
        return self.run("apk add -q " + packs, output=output, _try=_try, sudo=True)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return self.run("apk del -q " + packs, output=output, _try=_try, sudo=True)

#----------------------------------------------------------------------------------------------

class Setup(OnPlatform):
    def __init__(self, nop=False, verbose=False, sudo=True):
        OnPlatform.__init__(self)
        self.verbose = verbose
        self.nop = nop
        if verbose:
            self.runner = Runner(nop=nop, output=True)
        else:
            self.runner = Runner(nop=nop)
        self.stages = [0]
        self.platform = Platform()
        self.os = self.platform.os
        self.arch = self.platform.arch
        self.osnick = self.platform.osnick
        self.dist = self.platform.dist
        self.ver = self.platform.os_ver
        self.os_version = self.platform.os_version
        self.version = self.platform.version(full=True) # deprecated
        self.repo_refresh = True

        self.package_manager = PackageManager.detect(self.platform, self.runner)

        self.python = sys.executable
        os.environ["PYTHONWARNINGS"] = 'ignore:DEPRECATION::pip._internal.cli.base_command'

        self.sudoIf(sudo)

    def setup(self):
        if self.repo_refresh:
            self.package_manager.update()
            self.python = paella.sh("command -v python{VER}".format(VER=sys.version_info.major))

        try:
            gitver = sh("cd {} && git rev-parse --short HEAD".format(os.path.abspath(os.path.dirname(__file__))))
        except:
            gitver = "?"
        print("# readies version: {}".format(gitver))

        self.invoke()

    def run(self, cmd, at=None, output="on_error", nop=None, _try=False, sudo=False, echo=True):
        return self.runner.run(cmd, at=at, output=output, nop=nop, _try=_try, sudo=sudo, echo=echo)

    @staticmethod
    def has_command(cmd):
        return Runner.has_command(cmd)

    @property
    def profile_d(self):
        if self.os == 'macos':
            return os.path.abspath(os.path.join(os.path.expanduser('~'), ".profile.d"))
        else:
            return "/etc/profile.d"

    def cp_to_profile_d(self, file, as_file=None):
        if not os.path.isfile(file):
            raise Error("file not found: %s" % file)
        d = self.profile_d
        if as_file is None:
            as_file = os.path.basename(file)
        not_mac = self.os != 'macos'
        if not os.path.isdir(d):
            self.run('mkdir -p "{}"'.format(d), sudo=not_mac)
        self.run('cp "{FROM}" "{TO}"'.format(FROM=file, TO=os.path.join(d, as_file)), sudo=not_mac)

    def cat_to_profile_d(self, text, as_file=None):
        file = paella.tempfilepath()
        paella.fwrite(file, textwrap.dedent(text))
        d = self.profile_d
        if as_file is None:
            as_file = os.path.basename(file)
        not_mac = self.os != 'macos'
        if not os.path.isdir(d):
            self.run('mkdir -p "{}"'.format(d), sudo=not_mac)
        self.run('cp "{FROM}" "{TO}"'.format(FROM=file, TO=os.path.join(d, as_file)), sudo=not_mac)
        os.unlink(file)

    def sudoIf(self, sudo=True):
        if sudo:
            self.run("true", sudo=True, echo=False)

    #------------------------------------------------------------------------------------------

    def install(self, packs, group=False, output="on_error", _try=False, **kwargs):
        return self.package_manager.install(packs, group=group, output=output, _try=_try, **kwargs)

    def uninstall(self, packs, group=False, output="on_error", _try=False):
        return self.package_manager.uninstall(packs, group=group, output=output, _try=_try)

    def group_install(self, packs, output="on_error", _try=False):
        return self.install(packs, group=True, output=output, _try=_try)

    def add_repo(self, repo_url, repo="", _try=False):
        return self.package_manager.add_repo(repo_url, repo=repo, _try=_try)

    #------------------------------------------------------------------------------------------

    def pip(self, cmd, output="on_error", _try=False):
        return self.run(self.python + " -m pip --disable-pip-version-check " + cmd,
                        output=output, _try=_try, sudo=False)

    def pip_install(self, cmd, output="on_error", _try=False):
        pip_user = ''
        # if self.os == 'macos' and 'VIRTUAL_ENV' not in os.environ:
        if 'VIRTUAL_ENV' not in os.environ:
            pip_user = '--user '
        return self.run("{PYTHON} -m pip install --disable-pip-version-check {PIP_USER} {CMD}".
                        format(PYTHON=self.python, PIP_USER=pip_user, CMD=cmd),
                        output=output, _try=_try, sudo=False)

    def pip_uninstall(self, cmd, output="on_error", _try=False):
        return self.run("{PYTHON} -m pip uninstall --disable-pip-version-check -y {CMD} || true".
                        format(PYTHON=self.python, CMD=cmd),
                        output=output, _try=_try, sudo=False)

    #------------------------------------------------------------------------------------------

    def install_downloaders(self, _try=False):
        if self.os == 'linux':
            self.install("ca-certificates", _try=_try)
        if not (self.platform.is_redhat_compat() and self.platform.os_version[0] >= 9):
            # has curl-minimal which conflicts with curl
            self.install("curl", _try=_try)
        self.install("wget unzip", _try=_try)

    def install_git_lfs_on_linux(self, _try=False):
        if self.arch == 'x64':
            lfs_arch = 'amd64'
        elif self.arch == 'arm64v8':
            lfs_arch = 'arm64'
        elif self.arch == 'arm32v7':
            lfs_arch = 'arm'
        else:
            raise Error("Cannot determine platform for git-lfs installation")
        self.run("""
            set -e
            d=$(mktemp -d /tmp/git-lfs.XXXXXX)
            mkdir -p $d
            wget -q https://github.com/git-lfs/git-lfs/releases/download/v{LFS_VER}/git-lfs-linux-{ARCH}-v{LFS_VER}.tar.gz -O $d/git-lfs.tar.gz
            (cd $d; tar xf git-lfs.tar.gz)
            $d/install.sh
            rm -rf $d
            """.format(LFS_VER=GIT_LFS_VER, ARCH=lfs_arch), sudo=True)

    def install_gnu_utils(self, _try=False):
        packs = ""
        path = "/usr/local/bin"
        if self.os == 'macos':
            packs= "make coreutils findutils gnu-sed gnu-tar gawk gpatch"
            path = os.path.abspath(os.path.join(os.path.expanduser("~"), ".local", "bin"))
            if not os.path.isdir(path):
                os.makedirs(path)
        elif self.os == 'freebsd':
            packs = "gmake coreutils findutils gsed gtar gawk"
        self.install(packs)

        for x in ['make', 'find', 'xargs', 'sed', 'tar', 'mktemp', 'du']:
            dest = os.path.join(path, x)
            if not os.path.exists(dest):
                src = paella.sh("command -v g{}".format(x)).strip()
                if os.path.exists(dest):
                    os.unlink(dest)
                os.symlink(src, dest)
            else:
                eprint("Warning: {} exists - not replaced".format(dest))

        if self.os == 'macos':
            destfile = os.path.join(self.profile_d, 'readies-gnu-utils.sh')
            with open(destfile, 'w+') as fp:
                fp.write("export PATH={}:$PATH".format(path))

    def install_linux_gnu_tar(self, _try=False):
        if self.os != 'linux':
            eprint("Warning: not Linux - tar not installed")
            return
        if self.arch != 'x64':
            raise Error("Cannot install gnu tar on non-x64 platform")
        self.run("""
            dir=$(mktemp -d /tmp/tar.XXXXXX)
            (cd $dir; wget --no-verbose -O tar.tgz http://redismodules.s3.amazonaws.com/readies/gnu/gnu-tar-1.32-x64-centos7.tgz; tar -xzf tar.tgz -C /; )
            rm -rf $dir
            """, sudo=True)

    def setup_dotlocal(self):
        self.cat_to_profile_d(r'''
                if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                    export PATH="$HOME/.local/bin${PATH:+":$PATH"}"
                fi
            ''', "dotlocal.sh")

    # deprecated
    def install_ubuntu_modern_gcc(self, _try=False):
        return
