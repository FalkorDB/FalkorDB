function test194(tasks)
%TEST194 test GxB_Vector_diag

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test194 ----------- V = diag (A,k)\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {       'logical',        'logical', 10, 4, 0.5, 1, 1, -10}, ... % (  2,   2)
    {       'logical',        'logical', 10, 4, 0.5, 1, 1,  -2}, ... % (  1,   3)
    {       'logical',           'int8', 10, 4, 0.5, 1, 1, -10}, ... % (  1,   4)
    {       'logical',        'logical', 10, 4, 0.5, 1, 0, -10}, ... % (  1,   5)
    } ;
end

track_coverage = false ;
if (track_coverage)
    global GraphBLAS_grbcov
    track_coverage = ~isempty (GraphBLAS_grbcov) ;
    clast = sum (GraphBLAS_grbcov > 0) ;
    cfirst = clast ;
end

rng ('default') ;
GB_builtin_complex_set (true) ;

atype_last = '' ;
m_last = -1 ;
n_last = -1 ;
dd_last = -1 ;
A = [ ] ;

for kk = 1:length(tasks)
    task = tasks {kk} ;
    atype = task {1} ;
    vtype = task {2} ;
    m = task {3} ;
    n = task {4} ;
    dd = task {5} ;
    sparsity_control = task {6} ;
    csc = task {7} ;
    k = task {8} ;

    if (~(isequal (atype, atype_last) && m == m_last && n == n_last && ...
        dd == dd_last))
        A = GB_spec_random (m, n, dd, 128, atype) ;
        atype_last = atype ;
        m_last = m ;
        n_last = n ;
        dd_last = dd ;
    end

    A.sparsity = sparsity_control ;
    A.is_csc = csc ;
    V2 = GB_spec_vdiag (A, k, vtype) ;
    V1 = GB_mex_vdiag  (A, k, vtype) ;
    GB_spec_compare (V1, V2) ;

    if (track_coverage)
        c = sum (GraphBLAS_grbcov > 0) ;
        d = c - clast ;
        if (d > 0)
            aa = sprintf ('''%s''', atype) ;
            vv = sprintf ('''%s''', vtype) ;
            fprintf ('{%16s, %16s, %d, %d, %g, %d, %d, %3d},', ...
                aa, vv, m, n, dd, sparsity_control, csc, k) ;
            fprintf (' ... %% (%3d, %3d)\n', d, c-cfirst) ;
        end
        clast = c ;
    else
        fprintf ('.') ;
    end
end

fprintf ('\n') ;
fprintf ('test194: all tests passed\n') ;

