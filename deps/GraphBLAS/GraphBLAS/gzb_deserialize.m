function C = gzb_deserialize (ghb, blob)
%GZB_DESERIALIZE: wrapper for gbmex_deserialize mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    C = GhB (gbmex_deserialize (1, blob)) ;
else
    C = GrB (gbmex_deserialize (0, blob)) ;
end

