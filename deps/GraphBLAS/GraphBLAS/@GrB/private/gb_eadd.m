function C = gb_eadd (A, op, B)
%GB_EADD C = A+B, sparse matrix 'addition' using the given op.
% The pattern of C is the set union of A and B.  This method assumes the
% identity value of the op is zero.  That is, x+0 = x+0 = x.  The binary
% operator op is only applied to entries in the intersection of the
% pattern of A and B.
%
% The inputs A and B are built-in matrices or GraphBLAS structs (not GrB
% objects).  The result is a typically a GraphBLAS struct.
%
% See also GrB/plus, GrB/minus, GrB/bitxor, GrB/bitor, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[am, an, atype] = gbsize (A) ;
[bm, bn, btype] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
type = gboptype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars.  Result is also a scalar.
        C = gbeadd (A, op, B) ;
    else
        % A is a scalar, B is a matrix.  Result is full, unless A == 0.
        if (gb_scalar (A) == 0)
            % C = 0+B is a built-in matrix if B is a built-in matrix
            C = B ;
        else
            % expand A to a full matrix
            A = gb_scalar_to_full (bm, bn, type, gb_fmt (B), A) ;
            C = gbeadd (A, op, B) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar.  Result is full, unless B == 0.
        if (gb_scalar (B) == 0)
            % C = A+0 is a built-in matrix if A is a built-in matrix
            C = A ;
        else
            % expand B to a full matrix
            B = gb_scalar_to_full (am, an, type, gb_fmt (A), B) ;
            C = gbeadd (A, op, B) ;
        end
    else
        % both A and B are matrices.  Result is sparse.
        C = gbeadd (A, op, B) ;
    end
end

