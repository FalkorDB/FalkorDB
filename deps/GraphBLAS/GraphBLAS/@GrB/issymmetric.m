function s = issymmetric (G, option)
%ISSYMMETRIC Determine if a GraphBLAS matrix is real or complex symmetric.
% issymmetric (G) is true if G equals G.' and false otherwise.
% issymmetric (G, 'skew') is true if G equals -G.' and false otherwise.
% issymmetric (G, 'nonskew') is the same as issymmetric (G).
%
% See also GrB/ishermitian.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (nargin < 2)
    option = 'nonskew' ;
end

s = gb_issymmetric (G, option, false) ;

