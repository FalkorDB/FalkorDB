function [nvals, nzmax] = gzb_nvals (G)
%GZB_NVALS wrapper for gbmex_nvals.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

[nvals, nzmax] = gbmex_nvals (G) ;

