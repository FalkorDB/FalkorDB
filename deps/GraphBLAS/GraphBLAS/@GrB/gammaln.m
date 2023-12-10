function C = gammaln (G)
%GAMMALN logarithm of gamma function.
% C = gammaln (G) is the natural logarithm of the gamma function of each
% entry of G.  Since gammaln (0) = inf, the result is a full matrix.
% G must be real.
%
% See also GrB/gammaln.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('gammaln', gbfull (G, type))) ;

