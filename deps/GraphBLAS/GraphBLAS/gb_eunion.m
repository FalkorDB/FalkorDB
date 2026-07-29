function C = gb_eunion (ghb, A, op, B)
%GB_EUNION C = A+B, matrix 'addition' using the given op.  Not user-callable.
% The pattern of C is the set union of A and B.  Entries in A but not B, or in
% B but not A, are assumed to have the value zero.  The op is applied to all
% entries in the set union of the pattern of A and B.
%
% The inputs A and B are built-in matrices or GrB objects or structs.  The
% result a GraphBLAS struct.

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
        % A is a scalar, B is a matrix.  Result is full.
        % expand A to a full matrix
        a = gb_scalar_to_full (1, bm, bn, type, gb_fmt (B), A) ;
        C = gzb_eadd (ghb, a, op, B) ;
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar.  Result is full.
        % expand B to a full matrix
        b = gb_scalar_to_full (1, am, an, type, gb_fmt (A), B) ;
        C = gzb_eadd (ghb, A, op, b) ;
    else
        % both A and B are matrices.  Result is sparse.
        C = gzb_eunion (ghb, A, op, B) ;
    end
end

