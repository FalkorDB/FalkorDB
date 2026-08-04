function C = prune (G, id)
%GHB.PRUNE remove explicit values from a matrix.
% C = GhB.prune (G) removes any explicit zeros from G.
% C = GhB.prune (G, id) removes entries equal to the given scalar id.
%
% See also GhB/full, GhB.select, GhB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    C = gb_prune (1, G) ;
else
    C = gb_prune (1, G, id) ;
end

