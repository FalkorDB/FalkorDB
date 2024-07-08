#!/usr/bin/env python3

import sys
import os
import argparse

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
READIES = ROOT
sys.path.insert(0, ROOT)
import paella

#----------------------------------------------------------------------------------------------

class SystemSetup(paella.Setup):
    def __init__(self, args):
        paella.Setup.__init__(self, args.nop)

    def common_first(self):
        # self.install("")
        # self.group_install("")
        # self.setup_pip()
        # self.pip_install("")
        # self.run("")

        print(f"")
        print("# common-first")
        self.install_downloaders()
        self.setup_dotlocal()

    def linux_first(self):
        print("# linux-first")

    def debian_compat(self):
        print("# debian-compat")

    def redhat_compat(self):
        print("# redhat-compat")

    def archlinux(self):
        print("# archlinux")

    def fedora(self):
        print("# fedora")

    def linux_last(self):
        print("# linux-last")

    def macos(self):
        print("# macos")
        self.install_gnu_utils()

    def common_last(self):
        print("# common-last")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
# parser.add_argument('--bool', action="store_true", help="flag")
# parser.add_argument('--int', type=int, default=1, help='number')
# parser.add_argument('--str', type=str, default='str', help='string')
args = parser.parse_args()

BB()
SystemSetup(args).setup()
