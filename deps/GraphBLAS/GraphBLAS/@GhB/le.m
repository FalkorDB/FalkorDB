function C = le (A, B)
%A <= B less than or equal to.
% C = (A <= B) compares A and B element-by-element.  One or both may be
% scalars.  Otherwise, A and B must have the same size.
%
% See also GhB/lt, GhB/gt, GhB/ge, GhB/ne, GhB/eq.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_le (1, A, B) ;

