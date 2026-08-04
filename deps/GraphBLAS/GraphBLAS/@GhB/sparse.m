function C = sparse (G)
%SPARSE make a copy of a GraphBLAS sparse matrix.
% If G is already sparse, C = sparse (G) simply makes a copy of G.  If G is
% full or bitmap, C = sparse (G) returns C as sparse or hypersparse.  Explicit
% zeros are not removed.  To remove them use C = GhB.prune(G).
%
% See also GhB/issparse, GhB/full, GhB.type, GhB/prune, GhB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_sparse (1, G) ;

