function s = ishermitian (G, option)
%ISHERMITIAN Determine if a matrix is Hermitian or real symmetric.
% ishermitian (G) is true if G equals G' and false otherwise.
% ishermitian (G, 'skew') is true if G equals -G' and false otherwise.
% ishermitian (G, 'nonskew') is the same as ishermitian (G).
%
% See also GrB/issymmetric.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (nargin < 2)
    option = 'nonskew' ;
end

s = gb_issymmetric (G, option, true) ;

