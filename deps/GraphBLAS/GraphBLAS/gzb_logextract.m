function C = gzb_logextract (ghb, A, M)
%GZB_LOGEXTRACT: wrapper for gbmex_logextract mexFunction. Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    C = GhB (gbmex_logextract (1, A, M)) ;
else
    C = GrB (gbmex_logextract (0, A, M)) ;
end

