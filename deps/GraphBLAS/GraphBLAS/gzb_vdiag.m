function C = gzb_vdiag (ghb, A, k)
%GZB_VDIAG: wrapper for gbmex_vdiag mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    C = GhB (gbmex_vdiag (1, A, k)) ;
else
    C = GrB (gbmex_vdiag (0, A, k)) ;
end

