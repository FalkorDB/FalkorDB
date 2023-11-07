function C = gb_eunion (A, op, B)
%GB_EUNION C = A+B, sparse matrix 'addition' using the given op.
% The pattern of C is the set union of A and B.  Entries in A but not B,
% or in B but not A, are assumed to have the value zero.  The op is applied
% to all entries in the set union of the pattern of A and B.
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
        % A is a scalar, B is a matrix.  Result is full.
        % expand A to a full matrix
        A = gb_scalar_to_full (bm, bn, type, gb_fmt (B), A) ;
        C = gbeadd (A, op, B) ;
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar.  Result is full.
        % expand B to a full matrix
        B = gb_scalar_to_full (am, an, type, gb_fmt (A), B) ;
        C = gbeadd (A, op, B) ;
    else
        % both A and B are matrices.  Result is sparse.
        C = gbeunion (A, 0, op, B, 0) ;
    end
end

