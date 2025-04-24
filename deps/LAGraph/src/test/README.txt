To run the tests:

    cd LAGraph/build
    cmake .. ; make ; make test

LAGraph and GraphBLAS will be tested on many problems, and their
results checked against known results.  Memory faults are tested
via a set of "brutal" malloc/calloc/realloc/free methods, but this
requires SuiteSparse:GraphBLAS.

To run with test coverage:

    cd LAGraph/build
    cmake -DCOVERAGE=1 .. ; make ; make test_coverage

On the Mac, you should use gcc-11, from "brew install gcc".
You then must ensure your lcov command is from gcc-11, not clang.
Then do:

    cd LAGraph/build
    CC=gcc-11 CXX=g++-11 cmake -DCOVERAGE=1 .. ; make ; make test_coverage

To recompile the tests, do:

    cd LAGraph/build
    rm -rf *

and then compile the tests as shown above.
