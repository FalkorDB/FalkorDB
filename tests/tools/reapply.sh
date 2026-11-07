#!/bin/bash
# Regenerate the mechanical part of the flow migration from a clean HEAD.
# Manual edits (common.py, base.py, the Environment.skip sites) are re-applied
# on top so the tree ends up in the same state the PR commits.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/venv/lib/python3.14/site-packages${PYTHONPATH:+:$PYTHONPATH}"

keep=(tests/flow/common.py tests/flow/base.py)
for f in "${keep[@]}"; do cp "$f" "$f.keep"; done
git checkout -- tests/flow/
python3 tests/tools/rltest2pytest.py tests/flow
for f in "${keep[@]}"; do mv "$f.keep" "$f"; done

for f in test_intern_string test_replication test_replication_states test_ro_query; do
  python3 - "tests/flow/$f.py" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s2 = re.sub(r'Environment\.skip\(None\)\s*#\s*(.*)',
            lambda m: 'pytest.skip("%s")' % m.group(1).strip(), s)
open(p, 'w').write(s2)
PY
done

# testLoadTimeConfig.__init__ only called super().__init__(), which is a no-op
# (FlowTestsBase defines no constructor) and is invalid on a classmethod. The
# codemod refuses it rather than silently dropping the call.
python3 - <<'PY'
p = 'tests/flow/test_config.py'
s = open(p).read()
old = '''    def __init__(self):
        """
        Initialize the test class.

        Note: super().__init__() must be called to ensure FlowTestsBase
        performs any required setup (e.g. connecting to the test environment).
        """
        super().__init__()
'''
new = '''    @classmethod
    def setup_class(cls):
        """Nothing to set up: every test here launches its own environment."""
'''
assert old in s, 'test_config.py shape changed'
open(p, 'w').write(s.replace(old, new))
PY

python3 -W ignore -m py_compile tests/flow/*.py && echo "all files parse"
