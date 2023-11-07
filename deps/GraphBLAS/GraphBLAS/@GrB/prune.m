function C = prune (G, id)
%GRB.PRUNE remove explicit values from a matrix.
% C = GrB.prune (G) removes any explicit zeros from G.
% C = GrB.prune (G, id) removes entries equal to the given scalar id.
%
% See also GrB/full, GrB.select, GrB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    id = 0 ;
else
    id = gb_get_scalar (id) ;
end

if (isobject (G))
    % extract the contents of a GraphBLAS matrix
    G = G.opaque ;
end

if (id == 0)
    % prune zeros
    C = GrB (gbselect (G, 'nonzero')) ;
else
    % prune entries equal to id
    C = GrB (gbselect (G, '~=', id)) ;
end

