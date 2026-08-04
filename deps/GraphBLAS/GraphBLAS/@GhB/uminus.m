function C = uminus (G)
%UMINUS negate a matrix.
% C = -G negates the entries of the matrix G.
%
% See also GhB.apply, GhB/minus, GhB/uplus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gzb_apply (1, '-', G) ;

