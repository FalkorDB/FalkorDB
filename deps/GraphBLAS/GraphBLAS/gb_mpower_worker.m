function C = gb_mpower_worker (ghb, A, b)
%GB_MPOWER_WORKER C = A^b where b > 0 is an integer.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (b == 1)
    C = gb_dup (ghb, A) ;
else
    C = gb_mpower_worker (ghb, A, floor (b/2)) ;
    C = gzb_mtimes (ghb, C, C) ;
    if (mod (b, 2) == 1)
        C = gzb_mtimes (ghb, C, A) ;
    end
end

