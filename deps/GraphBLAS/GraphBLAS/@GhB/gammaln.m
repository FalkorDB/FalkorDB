function C = gammaln (G)
%GAMMALN logarithm of gamma function.
% C = gammaln (G) is the natural logarithm of the gamma function of each entry
% of G.  Since gammaln (0) = inf, the result is a full matrix.  G must be real.
%
% See also GhB/gammaln.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_gamma (1, 'gammaln', G) ;

