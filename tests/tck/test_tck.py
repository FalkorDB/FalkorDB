from RLTest import Env
from behave.__main__ import main as behave_main

# this test class is built in order to use 'behave' framework
# inside an RLTests managed environment.
# This class is instanciated in RLTest processes during runtime and
# the call to test_tck method is intializing and executing a 'behave'
# framework with the TCK feature tests inside the 'features' folder.
# See /features/steps/Steps.py and utils/graphs.py to for understanding how
# 'behave' step is preformed against RLTest environment


def test_tck():
    env = Env(decodeResponses=True)
    cmd = ["./features/", '--tags=-crash', '--tags=-skip', "--no-capture"]
    #  cmd = ["./features/", '--tags=-crash'] # Run all tests except crashing tests
    if not env.verbose:
        cmd.append('--format=progress')
    if env.getEnvKwargs()['debugger']:
        cmd.append('--tags=-leak')
    res = behave_main(cmd)
    res = 'pass' if res == 0 else 'fail'
    env.assertEquals(res, 'pass')
