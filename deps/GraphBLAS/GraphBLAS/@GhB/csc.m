function C = csc (G)
%CSC cosecant.
% C = csc (G) is the cosecant of each entry of G.  Since csc (0) is nonzero, C
% is a full matrix.
%
% See also GhB/acsc, GhB/csch, GhB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_csc (1, G) ;

