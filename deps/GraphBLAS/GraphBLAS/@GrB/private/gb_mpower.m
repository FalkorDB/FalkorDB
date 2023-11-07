function C = gb_mpower (A, b)
%GB_MPOWER C = A^b where b > 0 is an integer

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (b == 1)
    C = A ;
else
    T = gb_mpower (A, floor (b/2)) ;
    C = gbmxm (T, '+.*', T) ;
    clear T ;
    if (mod (b, 2) == 1)
        C = gbmxm (C, '+.*', A) ;
    end
end

