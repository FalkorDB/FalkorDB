function C = and (A, B)
%& logical AND.
% C = (A & B) is the element-by-element logical AND of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GhB/or, GhB/xor, GhB/not.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_and (1, A, B) ;

