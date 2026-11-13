#!/bin/bash
# m1opt/build.sh TAG: release+LTO test binary of the graph lib, copied to m1opt/bin/graph-test-TAG
S=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad
W=$S/m1opt
cd $W
export CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++
export CARGO_TARGET_DIR=$W/target-opt
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
cargo test --release -p graph --features index-falkordb --lib --no-run --message-format=json-render-diagnostics 2>$W/build-$1.err | tee $W/build-$1.json | python3 -c '
import sys,json
for l in sys.stdin:
    try: m=json.loads(l)
    except: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]=="graph": print(m["executable"])
' > $W/build-$1.exe
rc=${PIPESTATUS[0]}
EXE=$(tail -1 $W/build-$1.exe)
mkdir -p $W/bin
if [ $rc -eq 0 ] && [ -n "$EXE" ]; then cp "$EXE" $W/bin/graph-test-$1; echo "BUILT $W/bin/graph-test-$1 md5=$(md5 -q $W/bin/graph-test-$1) from $EXE"; else echo "BUILD FAILED rc=$rc"; tail -30 $W/build-$1.err; fi
