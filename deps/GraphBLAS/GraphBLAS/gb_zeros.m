function C = gb_zeros (ghb, varargin)
%GB_ZEROS implements GrB.zeros and GhB.zeros.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gb_parse_args ('zeros', varargin {:}) ;
C = gzb (ghb, m, n, type) ;

