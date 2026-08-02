function n = gb_length (G)
%GB_LENGTH implements GrB/length and GhB/length.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gb_size (G) ;

if (m == 0 || n == 0)
    n = 0 ;
else
    n = max (m, n) ;
end

