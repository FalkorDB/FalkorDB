function [m, n] = gb_size (A)
%GB_SIZE get the size of a matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end
[m, n] = gbmex_size (A) ;

