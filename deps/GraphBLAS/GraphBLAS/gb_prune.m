function C = gb_prune (ghb, G, id)
%GB_PRUNE implements GrB.prune and GhB.prune.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (nargin < 3)
    id = 0 ;
else
    id = gb_get_scalar (id) ;
end

if (id == 0)
    % prune zeros
    C = gzb_select (ghb, G, 'nonzero') ;
else
    % prune entries equal to id
    C = gzb_select (ghb, G, '~=', id) ;
end

