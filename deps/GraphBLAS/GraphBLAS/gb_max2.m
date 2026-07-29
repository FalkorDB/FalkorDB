function C = gb_max2 (ghb, op, A, B)
%GB_MAX2 2-input max.  Not user-callable.
% Implements C = max (A,B)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[am, an, atype] = gbmex_size (A) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gbmex_optype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars.  Result is also a scalar.
        C = gzb_eunion (ghb, A, op, B) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) > 0)
            % since A > 0, the result is full
            a = gb_scalar_to_full (1, bm, bn, ctype, gb_fmt (B), A) ;
            C = gzb_eadd (ghb, a, op, B) ;
        else
            % since A <= 0, the result is sparse.
            a = gzb_full (1, A) ;
            C = gzb_apply2 (ghb, a, op, B) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) > 0)
            % since B > 0, the result is full
            b = gb_scalar_to_full (1, am, an, ctype, gb_fmt (A), B) ;
            C = gzb_eadd (ghb, A, op, b) ;
        else
            % since B <= 0, the result is sparse.
            b = gzb_full (1, B) ;
            C = gzb_apply2 (ghb, A, op, b) ;
        end
    else
        % both A and B are matrices.  Result is sparse.
        C = gzb_eunion (ghb, A, op, B) ;
    end
end

