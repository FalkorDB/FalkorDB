function C = minus (A, B)
%MINUS sparse matrix subtraction, C = A-B.
% C = A-B subtracts the two matrices A and B.  If A and B are matrices,
% the pattern of C is the set union of A and B.  If one of A or B is a
% scalar, the scalar is expanded into a full matrix the size of the
% other matrix, and the result is a full matrix.
%
% See also GrB.eadd, GrB/plus, GrB/uminus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

C = GrB (gb_eunion (A, '-', B)) ;

