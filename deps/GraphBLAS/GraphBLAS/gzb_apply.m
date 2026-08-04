function C = gzb_apply (ghb, op, A, desc)
%GZB_APPLY: wrapper for gbmex_apply mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 4)
    desc = struct ;
end

if (ghb)
    C = GhB (gbmex_apply (1, op, A, desc)) ;
else
    C = GrB (gbmex_apply (0, op, A, desc)) ;
end

