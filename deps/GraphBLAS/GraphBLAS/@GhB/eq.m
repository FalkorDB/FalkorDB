function C = eq (A, B)
%A == B equal.
% C = (A == B) compares A and B element-by-element.  One or both may be
% scalars.  Otherwise, A and B must have the same size.
%
% The input matrices may be either GraphBLAS and/or built-in matrices, in any
% combination.  C is returned as a GraphBLAS GhB matrix.
%
% See also GhB/lt, GhB/le, GhB/gt, GhB/ge, GhB/ne.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_eq (1, A, B) ;

