function C = kron (A, B)
%KRON sparse Kronecker product.
% C = kron (A,B) is the sparse Kronecker tensor product of A and B.
%
% See also GrB.kronecker.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

C = GrB (gbkronecker (A, '*', B)) ;

