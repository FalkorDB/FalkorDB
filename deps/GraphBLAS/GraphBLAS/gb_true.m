function C = gb_true (ghb, varargin)
%GB_TRUE implements GrB.true and GhB.true.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, ~] = gb_parse_args ('true', varargin {:}) ;
C = gb_scalar_to_full (ghb, m, n, 'logical', gbmex_format, true) ;

