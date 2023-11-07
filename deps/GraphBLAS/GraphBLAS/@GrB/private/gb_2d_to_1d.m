function [k, mn] = gb_2d_to_1d (i, j, m, n)
%GB_2D_TO_1D convert 2D indices to 1D; the indices must be zero-based.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% check for overflow
if (double (m) * double (n) > 2^60)
    error ('GrB:error', 'problem too large') ;
end

% mn = the length of the vector x=A(:), if A is m by n
mn = int64 (m) * int64 (n) ;

% convert the 2D indices (i,j) into 1D indices (k)
k = i + j * m ;

