function C = mpower (A, B)
%A^B matrix power.
% C = A^B computes the matrix power of A raised to the B. A must be a
% square matrix.  B must an integer >= 0.
%
% See also GrB/power.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

[am, an, atype] = gbsize (A) ;
[bm, bn] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

if (a_is_scalar && b_is_scalar)
    C = GrB (gb_power (A, B)) ;
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
        C = GrB (gb_speye ('mpower', an, atype)) ;
    else
        % C = A^b where b > 0 is an integer
        C = GrB (gb_mpower (A, b)) ;
    end
end

