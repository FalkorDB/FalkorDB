# LAGraph/src/benchmark:  Demo programs that use LAGraph and GraphBLAS
# to run all of the GAP Benchmark problems.

LAGraph, (c) 2021-2023 by The LAGraph Contributors, All Rights Reserved.
SPDX-License-Identifier: BSD-2-Clause
See additional acknowledgments in the LICENSE file,
or contact permission@sei.cmu.edu for the full terms.

# NOTE: Please read this carefully first

BE SURE TO GET THE LATEST SuiteSparse:GraphBLAS.  LAGraph and SuiteSparse:
GraphBLAS are under intense development and even minor changes can have a big
impact on performance.  Also, please contact the authors of the library if you
have any questions about benchmarking.  (Tim Davis: davis@tamu.edu, in
particular).

If you run in vanilla mode, by compiling LAGraph with

    cmake -DLAGRAPH_VANILLA=1 ..

Then performance can be quite low since in this case LAGraph does not use
any SuiteSparse:GraphBLAS extensions.  We are still developing the
pure GrB implementations of these algorithms.

However, assuming things are stable, follow the instructions in the
next section.

# How to run the GAP benchmarks with LAGraph and SuiteSparse:GraphBLAS

To download the GAP benchmark matrices, optionally convert them to binary .grb
files (SuiteSparse-only), and to run the GAP benchmarks, do the following:

(1) Compile GraphBLAS (libgraphblas.so) and LAGraph (liblagraph.so), according
    to their respective instructions.

(2) Create a folder (say it's called "GAP") in a filesystem able to
    hold at least 400GB of data.

(3) Optionally: place your GAP folder in the same folder alongside the LAGraph
    and GraphBLAS folders, or create a symbolic link to the folder at that
    location, so the three directories exist side-by-side:

        LAGraph
        GraphBLAS
        GAP         can be a symbolic link to (say) /my/big/filestuff/GAP

    This step (3) is optional.  If you place your GAP folder somewhere else,
    say in /my/big/filestuff/GAP (or whatever your directory is) you will merely
    have to pass in the location to the do_gap_all script in Steps (6) and (7).

    To create a symbolic link (on Linux or the Mac), cd to the directory where
    you which to create the link, and type:

        ln -s /my/big/filestuff/GAP

(4) Go to http://sparse.tamu.edu/GAP and download the 5 matrices there in
    MatrixMarket format, into the GAP directory you chose in Step (2).

        GAP-twitter.tar.gz
        GAP-web.tar.gz
        GAP-road.tar.gz
        GAP-kron.tar.gz
        GAP-urand.tar.gz

(5) Uncompress all 5 files (in Linux: tar zvfx file.tar.gz).  You should now
    have the 10 following files in 5 folders:

        GAP-kron/GAP-kron.mtx
        GAP-kron/GAP-kron_sources.mtx
        GAP-road/GAP-road.mtx
        GAP-road/GAP-road_sources.mtx
        GAP-twitter/GAP-twitter.mtx
        GAP-twitter/GAP-twitter_sources.mtx
        GAP-urand/GAP-urand.mtx
        GAP-urand/GAP-urand_sources.mtx
        GAP-web/GAP-web.mtx
        GAP-web/GAP-web_sources.mtx

(6) If you wish to use the vanilla LAGraph + GraphBLAS to run the GAP
    benchmarks (it will be slow!), or if you wish to use LAGraph +
    SuiteSparse:GraphBLAS and don't mind the time it takes to load in a
    MatrixMarket file, then you can skip this Step (6).

    However, if you are using SuiteSparse:GraphBLAS, and wish to run the GAP
    benchmark multiple times, it is faster to create binary .grb files from the
    Matrix Market .mtx files.

    To create the binary files, in this directory (LAGraph/src/demo), type the
    following command:

        ./do_gap_binary

    or, if your GAP files are not located ../../../GAP relative to this
    directory, type the following instead where /my/big/filestuff/GAP is the
    directory you chose in Step (2) above.

        ./do_gap_binary /my/big/filestuff/GAP

    You should now have the 5 following additional files, and can run the GAP
    benchmarks using these binary files, instead of the MatrixMarket files.

        GAP-kron/GAP-kron.grb
        GAP-road/GAP-road.grb
        GAP-twitter/GAP-twitter.grb
        GAP-urand/GAP-urand.grb
        GAP-web/GAP-web.grb

(7) to run the GAP benchmarks with the original Matrix Market files (if you
    skipped Step (6), type the following command in this directory:

        ./do_gap_all > myoutput.txt

    To run the GAP benchmarks using the binary .grb files instead, type this:

        ./do_gap_all grb > myoutput.txt

    To run the benchmarks using a different directory than ../../../GAP,
    using the MatrixMarket .mtx files:

        ./do_gap_all mtx /my/big/filestuff/GAP > myoutput.txt

    To run the benchmarks using a different directory than ../../../GAP,
    using the binary .grb files:

        ./do_gap_all grb /my/big/filestuff/GAP > myoutput.txt

    As the do_gap_all script executes, you will see short summary of each test
    printed to stderr.  Detailed results are printed to stdout.  For the BFS,
    you will see 6 runs per matrix on stderr.  The default "black box" GAP
    benchmark results are the "parent only pushpull" results; the other results
    are for different problems (such as level-only, or parent+level), or with a
    different algorithm that is typically non-optimal (pushonly).  However, a
    pushonly BFS for the GAP-road graph is typically faster than the pushpull
    method, since the heuristic for push vs pull always selects the pull phase.

To control the number of OpenMP threads used in each demo, see the instructions
in bc_demo.c regarding NTHREAD_LIST.

You can also run these demos on the .mtx files in the LAGraph/data folder.
For example, to run the BFS on 64 randomly selected source nodes on the
bcsstk13.mtx, do:

    ../../build/src/demo/bfs_demo < ../data/bcsstk13.mtx

