function [x, y] = gb_get_2scalars (A)
%GB_GET_PAIR get a two scalars from a parameter of length 2.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gb_type (A) ;
desc.kind = 'full' ;
C = gb_builtin (gzb_full (1, A, type, 0, desc)) ;
x = C (1) ;
y = C (2) ;

