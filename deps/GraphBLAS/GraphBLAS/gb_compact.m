function [C, I, J] = gb_compact (ghb, A, id, symmetric)
%GB_COMPACT implements GrB.compact.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

symmetric = (nargin > 3 && isequal (symmetric, 'symmetric')) ;
if (symmetric)
    [m n] = gbmex_size (A) ;
    if (m ~= n)
        error ('A must be square to use the "symmetric" option') ;
    end
end

if (nargin > 2 && ~isempty (id))
    % prune identity values from A
    id = gb_get_scalar (id) ;
    if (id ~= 0)
        % prune a nonzero identity value from A
        [C, I, J] = gb_compact_worker (ghb, gzb_select (1, A, '~=', id), ...
            symmetric) ;
    elseif (~builtin ('issparse', A))
        % prune zeros from A
        [C, I, J] = gb_compact_worker (ghb, gzb_select (1, A, 'nonzero'), ...
            symmetric) ;
    else
        % compact A as-is
        [C, I, J] = gb_compact_worker (ghb, A, symmetric) ;
    end
else
    % compact A as-is
    [C, I, J] = gb_compact_worker (ghb, A, symmetric) ;
end

