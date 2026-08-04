function C = gb_mrdivide (ghb, A, B)
%GB_MRDIVIDE implements GrB/mrdivide and GhB/mrdivide.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isscalar (B))
    C = rdivide (A, B) ;
else
    C = gzb (ghb, builtin ('mrdivide', double (A), double (B))) ;
end

