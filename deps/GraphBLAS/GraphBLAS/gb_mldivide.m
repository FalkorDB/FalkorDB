function C = gb_mldivide (ghb, A, B)
%GB_MLDIVIDE implements GrB/mldivide and GhB/mldivide.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isscalar (A))
    C = rdivide (B, A) ;
else
    C = gzb (ghb, builtin ('mldivide', double (A), double (B))) ;
end

