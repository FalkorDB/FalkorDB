function C = log1p (G)
%LOG1P natural logarithm.
% C = log1p (G) is log(1+x) for each entry x of G.
% If any entry in G is < -1, the result is complex.
%
% See also GhB/log, GhB/log2, GhB/log10, GhB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_trig (1, 'log1p', G) ;

