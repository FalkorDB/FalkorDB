function s = numel (G)
%NUMEL the maximum number of entries in a matrix.
% numel (G) is m*n for the m-by-n GraphBLAS matrix G.  If m, n, or m*n
% exceed flintmax (2^53), the result is returned as a vpa symbolic value,
% to avoid integer overflow.
%
% See also GrB/nnz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
s = gb_numel (G) ;

