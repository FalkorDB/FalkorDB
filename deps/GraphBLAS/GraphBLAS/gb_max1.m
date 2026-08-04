function C = gb_max1 (ghb, op, A)
%GB_MAX1 single-input max.  Not user-callable.
% Implements C = max (A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbmex_size (A) ;
if (m == 1 || n == 1)
    % C = max (A) for a vector A results in a scalar C
    C = gb_maxall (ghb, op, A) ;
else
    C = gb_maxbycol (ghb, op, A) ;
end

