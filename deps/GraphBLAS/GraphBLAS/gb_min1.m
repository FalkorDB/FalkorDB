function C = gb_min1 (ghb, op, A)
%GB_MIN1 single-input min.  Not user-callable.
% Implements C = min (A)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbmex_size (A) ;
if (m == 1 || n == 1)
    % C = min (A) for a vector A results in a scalar C
    C = gb_minall (ghb, op, A) ;
else
    C = gb_minbycol (ghb, op, A) ;
end

