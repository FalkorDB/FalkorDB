function s = isfloat (G)
%ISFLOAT true for floating-point matrices.
% isfloat (G) is true if the matrix G has a type of 'double',
% 'single', 'single complex', or 'double complex'.
%
% See also GrB/isnumeric, GrB/isreal, GrB/isinteger, GrB/islogical,
% GrB.type, GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
s = gb_isfloat (gbtype (G)) ;

