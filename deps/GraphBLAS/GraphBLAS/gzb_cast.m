function C = gzb_cast (X, type)
%GZB_CAST: wrapper for gbmex_cast mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (X))
    X = struct (X) ;
end

C = GhB (gbmex_cast (X, type)) ;

