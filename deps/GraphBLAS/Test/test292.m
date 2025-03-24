function test292
%TEST292 test GxB_Vector_build_Scalar_Vector with a very large vector

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the amount of memory available
q = '''' ;
if (ismac)
    % Mac
    cmd = sprintf ('memory_pressure | awk %s/The system has/ { print $4}%s', q, q) ;
    [status, result] = system (cmd) ;
    memory = str2double (result) ;
elseif (ispc)
    % Windows: not supported
    memory = 0 ;
else
    % Linux
    cmd = sprintf ('cat /proc/meminfo | awk %s/MemTotal/ { print $2}%s', q, q) ;
    [status, result] = system (cmd) ;
    memory = str2double (result) * 1024 ;
end

% this test takes about 16*n bytes of memory
n = 5e9 ;
if (memory > n * 16)

    [nth chunk] = nthreads_get ;
    nthreads_set (8) ;
    GB_mex_burble (1) ;
    X = ones (n, 1, 'int8') ;
    s = int8 (3) ;
    op.opname = 'plus' ;
    op.optype = 'int8' ;
    desc.rowindex_list = 'use_indices' ;
    V = GB_mex_Vector_build (X, s, n, op, 'int8', desc) ;
    assert (nnz (V.matrix) == n)
    assert (size (V.matrix,1) == n) ;
    assert (min (V.matrix) == 3) ;
    assert (max (V.matrix) == 3) ;
    nthreads_set (nth, chunk) ;
    GB_mex_burble (0) ;
    fprintf ('test292: all tests passed\n') ;

else

    fprintf ('test292: skipped\n') ;

end
