function C = acsc (G)
%ACSC inverse cosecant.
% C = acsc (G) is the inverse cosecant of each entry of G.  Since acsc (0) is
% nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/csch, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_acsc (0, G) ;
