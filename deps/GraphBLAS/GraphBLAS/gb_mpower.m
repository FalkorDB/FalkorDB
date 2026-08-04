function C = gb_mpower (ghb, A, B)
%GB_MPOWER implements A^B for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

[am, an, atype] = gbmex_size (A) ;
[bm, bn] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

if (a_is_scalar && b_is_scalar)
    C = gb_power (ghb, A, B) ;
else
    if (am ~= an)
        error ('GrB:error', 'For C=A^B, A must be square') ;
    end
    if (~b_is_scalar)
        error ('GrB:error', ...
            'For C=A^B, B must be a non-negative integer scalar') ;
    end
    b = gb_scalar (B) ;
    if (~(isreal (b) && isfinite (b) && round (b) == b && b >= 0))
        error ('GrB:error', ...
            'For C=A^B, B must be a non-negative integer scalar') ;
    end
    if (b == 0)
        % C = A^0 = I
        if (isequal (atype, 'single complex'))
            atype = 'single' ;
        elseif (isequal (atype, 'double complex'))
            atype = 'double' ;
        end
        C = gb_speye (ghb, 'mpower', an, atype) ;
    else
        % C = A^b where b > 0 is an integer
        C = gb_mpower_worker (ghb, A, b) ;
    end
end

