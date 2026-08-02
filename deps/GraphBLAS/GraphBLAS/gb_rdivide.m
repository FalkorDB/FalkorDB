function C = gb_rdivide (ghb, A_arg, B)
%GB_RDIVIDE implements A./B for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A_arg))
    A_arg = struct (A_arg) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

[am, an, atype] = gbmex_size (A_arg) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gbmex_optype (atype, btype) ;

if (a_is_scalar && gb_scalar (A_arg) == 0 && gb_isfloat (ctype))
    A = 0 ;
else
    A = A_arg ;
end

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars
        b = gzb_full (1, B) ;
        C = gzb_emult (ghb, A, '/', b) ;
    else
        % A is a scalar, B is a matrix.
        % Expand B to full with type of C
        b = gzb_full (1, B, ctype) ;
        C = gzb_apply2 (ghb, A, '/', b) ;
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0 && gb_isfloat (atype))
            % 0/0 is Nan, and thus must be computed computed if A is
            % floating-point.  The result is a full matrix.
            % expand B into a full matrix and cast to the type of A
            b = gb_scalar_to_full (1, am, an, atype, gb_fmt (A), B) ;
            C = gzb_emult (ghb, A, '/', b) ;
        else
            % The scalar B is nonzero so just compute A/B in the pattern
            % of A.  The result is sparse (the pattern of A).
            C = gzb_apply2 (ghb, A, '/', B) ;
        end
    else
        % both A and B are matrices.  The result is a full matrix.
        a = gzb_full (1, A, ctype) ;
        b = gzb_full (1, B, ctype) ;
        C = gzb_emult (ghb, a, '/', b) ;
    end
end

