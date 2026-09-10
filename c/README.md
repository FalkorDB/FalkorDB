# The C consumer of the effects v3 conformance corpus

The fixtures at the root of this branch are the shared statement of the wire
format. This directory holds the **C** side's reader of them, kept here rather
than in the engine repo because the corpus is not a repo artifact — the ruling
was "we can always put the corpus in a side branch, including the tests. I don't
want corpus artifacts in the PR."

    c/tests/unit/effects_v3_corpus.h          the loader and the hash manifest
    c/tests/unit/test_effects_v3_corpus.c     corpus integrity, 7 tests
    c/tests/unit/test_effects_v3_roundtrip.c  decode/re-encode and truncation, 4 tests
    c/CORPUS-NOTES.md                         what the corpus pins, and what it does not

## This branch does not compile on its own, by design

The consumer includes `src/effects/effects_v3.h`, `src/globals.h` and
`src/util/thpool/pool.h`, and links the engine's objects. It is an auxiliary
library: it compiles when mounted into an engine tree, not standalone.

## Mounting it

```sh
git worktree add tests/fixtures/effects_v3 origin/effects-v3-corpus
cp -r tests/fixtures/effects_v3/c/tests/unit/. tests/unit/
cmake <build-dir> && make unit-tests          # reconfigure: the glob runs then
```

The copy and the reconfigure are the steps people forget, and forgetting them is
silent — `tests/unit/CMakeLists.txt` globs `test_*.c` at configure time, so an
uncopied consumer produces no target, and a target that does not exist does not
fail. The engine carries `tests/unit/test_effects_v3_wire.c` to notice: it
reports all four mount states, and `EFFECTS_V3_REQUIRE_CORPUS=1` makes an
incomplete mount fatal, which is what CI sets.

## What only this consumer can detect

The manifest in `effects_v3_corpus.h` hashes the fixtures against a table that
ships beside them, so it cannot detect a **stale** mirror — a mirror checked
against itself is consistent by construction. Measured: when the wire changed and
`rec_create_index` went from 96 bytes to 69, every hash matched the stale files
perfectly. What caught it was `test_effectsV3_decodesEveryFixture` handing the
bytes to the decoder and having it refuse them.

So a codec decoding these bytes is the only thing that detects staleness, and it
needs the corpus and a built engine in the same place. That is the whole reason
the mount exists rather than the fixtures being a static download.

## History

The detailed commit history of this consumer — 19 commits with the reasoning for
each check, each break-on-purpose verification, and the corrections along the way
— is on the engine repo's `backup/amber-loom-pre-split`, where it was written
before the split. It is worth reading before changing an assertion here.
