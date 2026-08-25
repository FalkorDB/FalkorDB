function C = gb_or (ghb, A, B)
%GB_OR implements A|B for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

[am, an, ~] = gbmex_size (A) ;
[bm, bn, ~] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;

if (a_is_scalar)
    if (b_is_scalar)
        % A and B are scalars
        C = gzb_emult (ghb, A, '|.logical', B) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) == 0)
            % A is false, so C is B typecasted to logical
            C = gzb (ghb, B, 'logical') ;
        else
            % A is true, so C is a full matrix the same size as B
            C = gb_scalar_to_full (ghb, bm, bn, 'logical', gb_fmt (B), true) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0)
            % B is false, so C is A typecasted to logical
            C = gzb (ghb, A, 'logical') ;
        else
            % B is true, so C is a full matrix the same size as A
            C = gb_scalar_to_full (ghb, am, an, 'logical', gb_fmt (A), true) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B
        C = gzb_eadd (ghb, A, '|.logical', B) ;
    end
end

