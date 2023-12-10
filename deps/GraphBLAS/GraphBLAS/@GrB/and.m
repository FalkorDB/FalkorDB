function C = and (A, B)
%& logical AND.
% C = (A & B) is the element-by-element logical AND of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/or, GrB/xor, GrB/not.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

[am, an] = gbsize (A) ;
[bm, bn] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

if (a_is_scalar)
    if (b_is_scalar)
        % A and B are scalars
        C = GrB (gbemult (A, '&.logical', B)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) == 0)
            % A is false, so C is empty, the same size as B
            C = GrB (gbnew (bm, bn, 'logical')) ;
        else
            % A is true, so C is B typecasted to logical
            C = GrB (gbnew (B, 'logical')) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0)
            % B is false, so C is empty, the same size as A
            C = GrB (gbnew (am, an, 'logical')) ;
        else
            % B is true, so C is A typecasted to logical
            C = GrB (gbnew (A, 'logical')) ;
        end
    else
        % both A and B are matrices.  C is the set intersection of A and B
        C = GrB (gbemult (A, '&.logical', B)) ;
    end
end

