function C = gb_eadd (ghb, A, op, B)
%GB_EADD C = A+B, matrix 'addition' (using an op).  Not user-callable.
% The pattern of C is the set union of A and B.  This method assumes the
% identity value of the op is zero.  That is, x+0 = 0+x = x.  The binary
% operator op is only applied to entries in the intersection of the
% pattern of A and B.
%
% The inputs A and B are built-in matrices or GraphBLAS structs (not GrB
% objects).  The result is a typically a GraphBLAS struct.
%
% See also GrB/plus, GrB/minus, GrB/bitxor, GrB/bitor, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

[am, an, atype] = gbmex_size (A) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
type = gbmex_optype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars.  Result is also a scalar.
        C = gzb_eadd (ghb, A, op, B) ;
    else
        % A is a scalar, B is a matrix.  Result is full, unless A == 0.
        if (gb_scalar (A) == 0)
            C = gb_dup (ghb, B) ;
        else
            % expand A to a full matrix
            a = gb_scalar_to_full (1, bm, bn, type, gb_fmt (B), A) ;
            C = gzb_eadd (ghb, a, op, B) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar.  Result is full, unless B == 0.
        if (gb_scalar (B) == 0)
            C = gb_dup (ghb, A) ;
        else
            % expand B to a full matrix
            b = gb_scalar_to_full (1, am, an, type, gb_fmt (A), B) ;
            C = gzb_eadd (ghb, A, op, b) ;
        end
    else
        % both A and B are matrices.  Result is sparse.
        C = gzb_eadd (ghb, A, op, B) ;
    end
end

