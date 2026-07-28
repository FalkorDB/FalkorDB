function [x,p] = gzb_argminmax (ghb, A, minmax, dim)
%GZB_ARGMINMAX: wrapper for gbmex_argminmax mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

[x, p] = gbmex_argminmax (ghb, A, minmax, dim) ;

if (ghb)
    x = GhB (x) ;
    p = GhB (p) ;
else
    x = GrB (x) ;
    p = GrB (p) ;
end

