function C = gzb_build (ghb, I, J, X, m, n, arg7, arg8, arg9)
%GZB_BUILD: wrapper for gbmex_build mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% The caller constructs I, J, and X as built-in matrices, so there is no need
% to check for any GrB input matrices to convert to their structs.

if (ghb)
    switch (nargin)
        case 7
            C = GhB (gbmex_build (1, I, J, X, m, n, arg7)) ;
        case 8
            C = GhB (gbmex_build (1, I, J, X, m, n, arg7, arg8)) ;
        case 9
            C = GhB (gbmex_build (1, I, J, X, m, n, arg7, arg8, arg9)) ;
    end
else
    switch (nargin)
        case 7
            C = GrB (gbmex_build (0, I, J, X, m, n, arg7)) ;
        case 8
            C = GrB (gbmex_build (0, I, J, X, m, n, arg7, arg8)) ;
        case 9
            C = GrB (gbmex_build (0, I, J, X, m, n, arg7, arg8, arg9)) ;
    end
end

