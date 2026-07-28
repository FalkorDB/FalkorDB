function C = kron (A, B)
%KRON sparse Kronecker product.
% C = kron (A,B) is the sparse Kronecker tensor product of A and B.
%
% See also GhB.kronecker.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gzb_kronecker (1, A, '*', B) ;

