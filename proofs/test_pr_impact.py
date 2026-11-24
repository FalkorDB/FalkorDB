#!/usr/bin/env python3
"""Tests for pr_impact.py: which fn owns a diff line.

  python3 proofs/test_pr_impact.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr_impact  # noqa: E402

SRC = textwrap.dedent('''\
    use std::fmt;

    /// Doc of a.
    pub fn a(x: u8) -> u8 {
        x + 1
    }

    #[allow(
        clippy::too_many_lines,
        reason = "long"
    )]
    // plain comment
    /* block
       comment */
    #[inline]
    pub fn b() {}

    pub fn c() {
        let v = [1, 2];
        fn nested() {}
        v.len();
    }
    ''').split('\n')


def fns_in(lines):
    return [(n, m[1], False) for n, l in enumerate(lines, 1)
            if (m := re.search(r'\bfn\s+(\w+)', l))]


class Spans(unittest.TestCase):
    def setUp(self):
        self.fns = fns_in(SRC)
        self.ends = pr_impact.ends_of(self.fns, SRC)

    def own(self, line):
        i = pr_impact.owner(self.fns, self.ends, line)
        return None if i is None else self.fns[i][1]

    def test_body_and_closing_brace_belong_to_the_fn(self):
        self.assertEqual(self.own(4), 'a')   # pub fn a
        self.assertEqual(self.own(5), 'a')   # body
        self.assertEqual(self.own(6), 'a')   # closing brace

    def test_lines_above_the_next_fn_belong_to_neither(self):
        # the blank line, multi-line #[allow(..)], comments and #[inline] above b
        for line in range(7, 16):
            self.assertIsNone(self.own(line), f'line {line}: {SRC[line - 1]!r}')
        self.assertEqual(self.own(16), 'b')

    def test_doc_comment_above_the_first_fn_belongs_to_neither(self):
        self.assertIsNone(self.own(3))

    def test_nested_fn_does_not_hide_code_before_it(self):
        self.assertEqual(self.own(19), 'c')  # let v = [1, 2];
        self.assertEqual(self.own(20), 'nested')

    def test_unclosed_bracket_is_not_an_attribute(self):
        lines = ['fn a() {', '    let x = [', '        1]', '    fn b() {}', '}']
        fns = fns_in(lines)
        ends = pr_impact.ends_of(fns, lines)
        self.assertEqual(pr_impact.owner(fns, ends, 3), 0)

    def test_code_ending_in_a_block_comment_is_kept(self):
        lines = ['fn a() {', '    g(); /* why */', '    fn b() {}', '}']
        fns = fns_in(lines)
        ends = pr_impact.ends_of(fns, lines)
        self.assertEqual(pr_impact.owner(fns, ends, 2), 0)


@unittest.skipUnless(shutil.which('git') and shutil.which('tar') and shutil.which('python3'),
                     'needs git, tar and python3')
class Analyse(unittest.TestCase):
    """pr_impact.analyse() on a two-commit repo, with `a` PROVEN."""

    BASE = textwrap.dedent('''\
        pub fn a(x: u8) -> u8 {
            x + 1
        }

        pub fn b(x: u8) -> u8 {
            x * 2
        }
        ''')

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.repo = os.path.join(self.dir, 'repo')
        self.proofs = os.path.join(self.dir, 'proofs')
        os.makedirs(os.path.join(self.repo, 'graph', 'src'))
        os.makedirs(os.path.join(self.repo, 'src'))
        os.makedirs(os.path.join(self.proofs, 'p'))
        shutil.copy(os.path.join(pr_impact.HERE, 'coverage.sh'), self.proofs)
        with open(os.path.join(self.proofs, 'p', 'COVERAGE.tsv'), 'w') as f:
            f.write('file\tfunction\tline\tbucket\tlean\n')
            f.write('graph/src/lib.rs\ta\t1\tPROVEN\tP.a_spec\n')
            f.write('graph/src/lib.rs\tb\t5\tNOT COVERED\t\n')
        self.git('init', '-q')
        self.write('src/lib.rs', '')
        self.base = self.commit(self.BASE)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def git(self, *args):
        return subprocess.run(['git', '-C', self.repo, '-c', 'user.name=t', '-c', 'user.email=t@t',
                               *args], check=True, capture_output=True, text=True).stdout.strip()

    def write(self, rel, text):
        with open(os.path.join(self.repo, rel), 'w', newline='\n') as f:
            f.write(text)

    def commit(self, text):
        self.write('graph/src/lib.rs', text)
        self.git('add', '-A')
        self.git('commit', '-q', '-m', 'c')
        return self.git('rev-parse', 'HEAD')

    def changed(self, text):
        head = self.commit(text)
        res = pr_impact.analyse(self.repo, self.base, head, self.proofs)
        return [c['fn'] for c in res['changed']]

    def test_attribute_above_the_next_fn_changes_nothing(self):
        head = self.BASE.replace('pub fn b', '#[allow(clippy::too_many_lines, reason = "x")]\npub fn b')
        self.assertEqual(self.changed(head), [])

    def test_doc_comment_above_the_next_fn_changes_nothing(self):
        self.assertEqual(self.changed(self.BASE.replace('pub fn b', '/// Doubles.\npub fn b')), [])

    def test_body_edit_is_still_reported(self):
        self.assertEqual(self.changed(self.BASE.replace('x + 1', 'x + 2')), ['a'])


if __name__ == '__main__':
    unittest.main()
