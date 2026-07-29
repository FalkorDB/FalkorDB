function C = gb_flip (ghb, A, dim)
%GB_FLIP implements flip for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

[m, n] = gbmex_size (A) ;

if (nargin == 2)
    if (m == 1)
        dim = 2 ;
    else
        dim = 1 ;
    end
else
    dim = gb_get_scalar (dim) ;
end

dim = floor (double (dim)) ;
if (dim <= 0)
    error ('GrB:error', 'dim must be positive') ;
end

if (dim == 1 && m ~= 1)
    % C = A (m:-1:1, :)
    C = gzb_extract (ghb, A, {m,-1,1}, { }) ;
elseif (dim == 2 && n ~= 1)
    % C = A (:, n:-1:1)
    C = gzb_extract (ghb, A, { }, {n,-1,1}) ;
else
    % nothing to do
    % C = A
    C = gb_dup (ghb, A) ;
end

