function C = ne (A, B)
%A ~= B not equal.
% C = (A ~= B) compares A and B element-by-element.  One or both may be
% scalars.  Otherwise, A and B must have the same size.
%
% See also GhB/lt, GhB/le, GhB/gt, GhB/ge, GhB/eq.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_ne (1, A, B) ;

