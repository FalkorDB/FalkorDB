function C = transpose (G)
%TRANSPOSE C = G.', array transpose.
% C = G.' is the array transpose of G.
%
% See also GrB.trans, GrB/ctranspose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
C = GrB (gbtrans (G)) ;

