function C = lt (A, B)
%A < B less than.
% C = (A < B) compares A and B element-by-element.  One or both may be scalars.
% Otherwise, A and B must have the same size.
%
% See also GrB/le, GrB/gt, GrB/ge, GrB/ne, GrB/eq.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_lt (0, A, B) ;

