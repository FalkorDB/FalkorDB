function C = gzb_extract (ghb, A, I, J)
%GZB_EXTRACT: wrapper for gbmex_extract mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    C = GhB (gbmex_extract (1, A, I, J)) ;
else
    C = GrB (gbmex_extract (0, A, I, J)) ;
end

