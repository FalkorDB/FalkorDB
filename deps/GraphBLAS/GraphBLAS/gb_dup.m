function C = gb_dup (ghb, A)
%GB_DUP make a copy of a GraphBLAS matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

C = gzb (ghb, gbmex_new (ghb, A)) ;

