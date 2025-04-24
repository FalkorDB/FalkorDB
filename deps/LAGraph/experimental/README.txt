--------------------------------------------------------------------------------
LAGraph/experimental/README.txt
--------------------------------------------------------------------------------

This experimental folder includes algorithms and utilities in various
stages of completion.  Their prototypes appear in LAGraph/include/LAGraphX.h.

When new methods methods are developed for LAGraph, they are first placed in
this folder, and then when they are finished and polised, they are moved into
the src/ folder.

--------------------------------------------------------------------------------
FUTURE WORK:

FUTURE: LAGraph_SingleSourceShortestPath:  Write a Basic method that
            computes G->emin (that's easy).  And G->emax so it can decide on
            Delta on its own (hard to do).

FUTURE: LAGraph_BreadthFirstSearch basic method that computes G->AT
        and G->out_degree first.

FUTURE: all algorithms can use Basic and Advanced variants.

FUTURE: file I/O with a common, binary, non-opaque format that all
            GraphBLAS libraries can read.  Ideally using compression
            (see SuiteSparse:GraphBLAS uses LZ4 for its GrB_Matrix_serialize/
            deserialize and it would not be hard to add LZ4 to LAGraph).
            We could add the format to the *.lagraph file format now used
            by the experimental/utility/LAGraph_S*.c methods.  That is, the
            same file format *.lagraph could include matrices in either
            the universal format, or a library-specific format.  The format
            would be specified by the json header in the *.lagraph file.
            See the doc/lagraph_format.txt file for details.

            It would require the LAGraph_SLoad, *SSave, etc to move from
            experimental to src, and those functions need work (API design,
            text compression).  Adding this feature also requires an
            LAGraph_serialize/deserialize, which requies LZ4 compression.  We
            could do this for v1.1.

FUTURE: can we exploit asynch algorithms? C += A*C for example?
        Gauss-Seidel, Afforest, etc?  I can do it in GraphBLAS; can LAGraph
        use it?

FUTURE: for LG_CC_FastSV6.c:

    * need new GxB methods in GraphBLAS for CC,
        GxB_select with GxB_RankUnaryOp, and GxB_extract with GrB_Vectors as
        inputs instead of (GrB_Index *) arrays.

FUTURE: add interfaces to external packages.

    GTgraph: (Madduri and Bader) for generating synthetic graphs
    CSparse or CXSparse (for depth-first search, scc, dmperm, amd,
        in the sequential case)
    graph partitioning: METIS, Mongoose, etc
    SuiteSparse solvers (UMFPACK, CHOLMOD, KLU, SPQR, ...)
    graph drawing methods
    others?

FUTURE: interfaces to MATLAB, Python, Julia, etc.

FUTURE: need more algorithms and utilities

