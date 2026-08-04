function C = gb_emult (ghb, A, op, B)
%GB_EMULT C = A.*B, matrix element-wise multiplication.  Not user-callable.
% C = gb_emult (A, op, B) computes the element-wise multiplication of A and B
% using the operator op, where the op is '*' for C=A.*B.  If both A and B are
% matrices, the pattern of C is the intersection of A and B.  If one is a
% scalar, the pattern of C is the same as the pattern of the one matrix.
%
% The input matrices may be either GraphBLAS structs and/or built-in matrices,
% in any combination.  C is returned as a GraphBLAS GrB or GhB matrix.

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
        % both A and B are scalars
        C = gzb_emult (ghb, A, op, B) ;
    else
        % A is a scalar, B is a matrix
        C = gzb_apply2 (ghb, gzb_full (1, A), op, B) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        C = gzb_apply2 (ghb, A, op, gzb_full (1, B)) ;
    else
        % both A and B are matrices
        C = gzb_emult (ghb, A, op, B) ;
    end
end

