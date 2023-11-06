function e = nzmax (G)
%NZMAX maximum number of entries in a matrix.
% Since the GraphBLAS data structure is opaque, nzmax (G) has no
% particular meaning.  Thus, nzmax (G) is simply max (GrB.entries (G), 1).
% It appears as an overloaded operator for a GrB matrix simply for
% compatibility with built-in sparse matrices.
%
% See also GrB/nnz, GrB.entries, GrB.nonz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
e = max (gbnvals (G), 1) ;

