function C = rdivide (A, B)
%RDIVIDE C = A./B, sparse matrix element-wise division.
% C = A./B when B is a matrix results in a full matrix C, with all
% entries present.  If A is a matrix and B is a scalar, then C has the
% pattern of A, except if B is zero and A is double, single, or complex.
% In that case, since 0/0 is NaN, C is a full matrix.
%
% See also GrB/ldivide, GrB.emult, GrB.eadd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

[am, an, atype] = gbsize (A) ;
[bm, bn, btype] = gbsize (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gboptype (atype, btype) ;

if (a_is_scalar && gb_scalar (A) == 0 && gb_isfloat (ctype))
    A = 0 ;
end

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars
        C = GrB (gbemult (A, '/', gbfull (B))) ;
    else
        % A is a scalar, B is a matrix.
        % Expand B to full with type of C
        C = GrB (gbapply2 (A, '/', gbfull (B, ctype))) ;
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0 && gb_isfloat (atype))
            % 0/0 is Nan, and thus must be computed computed if A is
            % floating-point.  The result is a full matrix.
            % expand B into a full matrix and cast to the type of A
            B = gb_scalar_to_full (am, an, atype, gb_fmt (A), B) ;
            C = GrB (gbemult (A, '/', B)) ;
        else
            % The scalar B is nonzero so just compute A/B in the pattern
            % of A.  The result is sparse (the pattern of A).
            C = GrB (gbapply2 (A, '/', B)) ;
        end
    else
        % both A and B are matrices.  The result is a full matrix.
        C = GrB (gbemult (gbfull (A, ctype), '/', gbfull (B, ctype))) ;
    end
end

