function s = gb_is_grb (A)
%GB_IS_GRB determine if a matrix is a GrB object.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = isequal (class (A), 'GrB') ;

