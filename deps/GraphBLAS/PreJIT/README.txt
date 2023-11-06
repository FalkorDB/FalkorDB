GraphBLAS/PreJIT:  This folder is empty in the GraphBLAS distribution.

JIT kernel source files created by GraphBLAS may be placed in this folder by
the end user, by copying them from your ~/.SuiteSparse/GrB*.*.* folder.
If GraphBLAS is then recompiled via cmake, the build system will compile all of
these kernels and make them available as 'pre-compiled JIT kernels'.  The
kernels are no longer JIT kernels since they are not compiled at run-time, but
they are still refered to as "JIT" kernels since they were at one time created
at run time by the GraphBLAS JIT.  Thus the name of this folder:
GraphBLAS/PreJIT.

If the GraphBLAS version is changed at all (even in the last digit), all *.c
files in this folder must be deleted.

If a user-defined type or operator is changed, the relevant kernels must also
be deleted.  For example, the GraphBLAS/Demo/Program/gauss_demo.c program
creates a user-defined gauss type, and two operators, addgauss and multgauss.
If the type and/or operators are changed, then the *gauss*.c files in this
folder should be deleted, so that the JIT can recompile them with the new
definition.  Otherwise, GraphBLAS will detect that the definitions do not
match, and it will not use the related kernels in this file.  Instead, it will
compile new ones in your ~/.SuiteSparse/GrB*.*.* folder.  GraphBLAS will
work with stale functions in this folder.  It will compile them and include
them in libgraphblas.so, but not use them at all.  So deleting them will reduce
the size of the compiled libgraphblas.so library.

These PreJIT kernels are also compiled into libgraphblas_matlab.so for use
in MATLAB.  Any JIT or PreJIT kernels used or created by libgraphblas.so
or libgraphblas_matlab.so can be used by each other, interchangebly.

