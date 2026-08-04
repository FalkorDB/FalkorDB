function C = gb_false (ghb, varargin)
%GB_FALSE implements GrB.false and GhB.false.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, ~] = gb_parse_args ('false', varargin {:}) ;
C = gzb (ghb, m, n, 'logical') ;

