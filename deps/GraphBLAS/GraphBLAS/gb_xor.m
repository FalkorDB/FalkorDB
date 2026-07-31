function C = gb_xor (ghb, A, B)
%GB_XOR implements xor for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % A and B are scalars
        C = gzb_emult (ghb, A, 'xor.logical', B) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) == 0)
            % A is false, so C is B typecasted to logical
            C = gzb (ghb, B, 'logical') ;
        else
            % A is true, so C is a full matrix the same size as B
            b = gzb_full (1, B, 'logical') ;
            C = gzb_apply (ghb, '~', b) ;
        end
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0)
            % B is false, so C is A typecasted to logical
            C = gzb (ghb, A, 'logical') ;
        else
            % B is true, so C is a full matrix the same size as A
            a = gzb_full (1, A, 'logical') ;
            C = gzb_apply (ghb, '~', a) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B
        C = gzb_eadd (ghb, A, 'xor.logical', B) ;
    end
end

