function C = or (A, B)
%| logical OR.
% C = (A | B) is the element-by-element logical OR of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/and, GrB/xor, GrB/not.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

[am, an, ~] = gbsize (A) ;
[bm, bn, ~] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

if (a_is_scalar)
    if (b_is_scalar)
        % A and B are scalars
        C = GrB (gbemult (A, '|.logical', B)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) == 0)
            % A is false, so C is B typecasted to logical
            C = GrB (gbnew (B, 'logical')) ;
        else
            % A is true, so C is a full matrix the same size as B
            C = GrB (gb_scalar_to_full (bm, bn, 'logical', gb_fmt (B), true)) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0)
            % B is false, so C is A typecasted to logical
            C = GrB (A, 'logical') ;
        else
            % B is true, so C is a full matrix the same size as A
            C = GrB (gb_scalar_to_full (am, an, 'logical', gb_fmt (A), true)) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B
        C = GrB (gbeadd (A, '|.logical', B)) ;
    end
end

