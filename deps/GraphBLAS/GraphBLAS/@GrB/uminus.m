function C = uminus (G)
%UMINUS negate a matrix.
% C = -G negates the entries of the matrix G.
%
% See also GrB.apply, GrB/minus, GrB/uplus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
C = GrB (gbapply ('-', G)) ;

