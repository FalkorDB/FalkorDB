function x = gb_get_scalar (a)
%GB_GET_SCALAR get a scalar from a matrix.  Not user-callable.
% The input can be builtin, GrB, or GhB.
% Returns an error if the input is not a scalar.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (a))
    a = struct (a) ;
end

[m, n] = gbmex_size (a) ;
if (m ~= 1 || n ~= 1)
    error ('GrB:error', 'input parameter %s must be a scalar', inputname (1)) ;
end

x = gb_scalar (a) ;

