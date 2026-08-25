function e = nzmax (G)
%NZMAX maximum number of entries in a matrix.
%
% See also GrB/nnz, GrB.entries, GrB.nonz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% nzmax (G) for a GrB or GhB G includes the # of entries held in the matrix,
% including zombies for a GhB matrix (which never appear in a GrB matrix).
% bitmap matrices return the same result as full matrices.  nzmax (G) includes
% the size of the G->Pending->[i,j,x] arrays for pending tuples in a GhB
% matrix.

[~, e] = gzb_nvals (G) ;

