function C = gb_ones (ghb, varargin)
%GB_ONES implements GrB.ones and GhB.ones.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gb_parse_args ('ones', varargin {:}) ;
C = gb_scalar_to_full (ghb, m, n, type, gbmex_format, 1) ;

