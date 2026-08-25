function x = gb_scalar (a)
%GB_SCALAR get contents of a scalar.  Not user-callable.
% The scalar a may be a built-in scalar or a GraphBLAS scalar.  Returns the
% result x as a built-in non-sparse scalar.  If the scalar has no entry (the
% built-in sparse(0)), then x is returned as zero.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (a))
    a = struct (a) ;
end

gbmex_wait (a) ;
[~, ~, x] = gbmex_extracttuples (1, a) ;
if (isempty (x))
    x = 0 ;
else
    x = x (1) ;
end

