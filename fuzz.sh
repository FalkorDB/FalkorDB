# build with fuzz feature
cargo build --features fuzz

# create directories for fuzzing
mkdir -p fuzz/corpus/fuzz_target_runtime/

# run tck tests to generate the corpus
TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s

# run fuzz test
CARGO_PROFILE_RELEASE_LTO=false cargo fuzz run fuzz_target_runtime -- -max_total_time=30

# minimize the corpus
CARGO_PROFILE_RELEASE_LTO=false cargo fuzz cmin fuzz_target_runtime

# generate coverage data
CARGO_PROFILE_RELEASE_LTO=false cargo fuzz coverage fuzz_target_runtime 

# generate the coverage report
# if os is linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # generate the coverage report for linux
    llvm-cov show -format=html -instr-profile=fuzz/coverage/fuzz_target_runtime/coverage.profdata -ignore-filename-regex=\.cargo/registry target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/fuzz_target_runtime > cov.html
fi

#  if os is darwin
if [[ "$OSTYPE" == "darwin"* ]]; then
    # generate the coverage report for darwin
    llvm-cov show -format=html -instr-profile=fuzz/coverage/fuzz_target_runtime/coverage.profdata -ignore-filename-regex=\.cargo/registry target/aarch64-apple-darwin/coverage/aarch64-apple-darwin/release/fuzz_target_runtime > cov.html
fi