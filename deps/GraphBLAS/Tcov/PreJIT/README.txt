These JIT kernels are created by test145.
They are placed here to use as PreJIT kernels for the Tcov tests.

This file has an intentionally stale function definition:

    GB_jit__AxB_dot2__0004000bbb0bbbcd__plus_my_rdiv.c

This file has an intentionally stale GraphBLAS version:

    GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2.c

These files should be valid PreJIT kernels:

    GB_jit__AxB_dot2__0004000bba0bbacf__plus_my_rdiv2.c
    GB_jit__AxB_dot2__0004015bbb0bbbcd.c
    GB_jit__AxB_dot2__0004100bba0baacf__plus_my_rdiv2.c
    GB_jit__AxB_dot2__0004100bba0babcd__plus_my_rdiv2.c
    GB_jit__AxB_dot2__0004100bba0babcf__plus_my_rdiv2.c
    GB_jit__AxB_dot2__0004100bba0bbac7__plus_my_rdiv2.c
    GB_jit__user_op__0__my_rdiv.c

This file will contain an index of the kernels listed above:

    GB_prejit.c

These files are created by test145 but must not be added as PreJIt kernels
(adding them will reduce test coverage of GB_AxB_dot4.c):

    GB_jit__AxB_dot4__0004014bbb0bbbc5.c
    GB_jit__AxB_dot4__0004014bbb0bbbcd.c
    GB_jit__AxB_dot4__0004014bbb0bbbcf.c
    GB_jit__AxB_dot4__0004900bba0baacf__plus_my_rdiv2.c
    GB_jit__AxB_dot4__0004900bba0babcf__plus_my_rdiv2.c
    GB_jit__AxB_dot4__0004900bba0bbac7__plus_my_rdiv2.c

If GraphBLAS is modified, the JIT kernels can be recreated as follows:

    (1) Run Test/test145 to create these JIT kernels, using the Test/test145.m
    script in the GraphBLAS/Test folder.

    (2) Copy the *dot2* and *user* JIT files from ~/.SuiteSparse/GrB*/c/, and
    place them in GraphBLAS/PreJIT.  Leave the *dot4* files.

    (3) Rebuild GraphBLAS.

    (4) Move the JIT files from GraphBLAS/PreJIT to GraphBLAS/Tcov/PreJIT, and
    move the GraphBLAS/Config/GB_prejit.c file to GraphBLAS/Tcov/PreJIT.

    (5) Modify Test/GB_mex_rdiv.c to trigger the stale PreJIT kernel case, by
    changing the string MY_RDIV.  After modifying it, rerun test145 and copy
    the final GB_jit__user_op__0__my_rdiv.c from your ~/.SuiteSparse folder to
    GraphBLAS/Tcov/PreJIT.

    (6) Edit one of the JIT kernels to trigger a stale GraphBLAS version;
        ( GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2.c ).

    (7) Edit the remaining JIT kernels so that they track whatever the current
    GraphBLAS version is.

    (8) Rerun the test coverage, using grbcov.m in GraphBLAS/Tcov, in MATLAB.

