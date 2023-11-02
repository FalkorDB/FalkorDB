function C = sparse (G)
%SPARSE make a copy of a GraphBLAS sparse matrix.
% If G is already sparse, C = sparse (G) simply makes a copy of G.
% If G is full or bitmap, C = sparse (G) returns C as sparse or hypersparse.
% Explicit zeros are not removed.  To remove them use C = GrB.prune(G).
%
% See also GrB/issparse, GrB/full, GrB.type, GrB/prune, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, sparsity] = gbformat (G.opaque) ;

switch (sparsity)
    case { 'hypersparse', 'sparse' }
        % nothing to do; G is already sparse or hypersparse
        C = G ;
    case { 'bitmap', 'full' }
        % convert G to sparse or hypersparse
        C = GrB (G, 'sparse/hypersparse') ;
end

