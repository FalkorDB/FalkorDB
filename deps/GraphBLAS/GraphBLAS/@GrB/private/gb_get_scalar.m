function x = gb_get_scalar (A)
%GB_GET_SCALAR get a scalar from a matrix

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

[m, n] = gbsize (A) ;
if (m ~= 1 || n ~= 1)
    error ('GrB:error', 'input parameter %s must be a scalar', inputname (1)) ;
end

x = gb_scalar (A) ;

