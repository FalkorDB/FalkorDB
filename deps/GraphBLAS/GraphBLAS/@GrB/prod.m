function C = prod (G, option)
%PROD product of elements.
% C = prod (G), where G is an m-by-n matrix, is a 1-by-n row vector C where
% C(j) is the product of all entries in G(:,j).  If G is a row or column
% vector, then prod (G) is a scalar product of all the entries in the
% vector.
%
% C = prod (G,'all') takes the product of all elements of G to a single
% scalar.
%
% C = prod (G,1) is the default when G is a matrix, which is to take the
% product of each column, giving a 1-by-n row vector.  If G is already a
% row vector, then C = G.
%
% C = prod (G,2) takes the product of each row, resulting in an m-by-1
% column vector C where C(i) is the product of all entries in G(i,:).
%
% The built-in prod (A, ... type, nanflag) allows for different kinds of
% products to be computed, and the NaN behavior can be specified.  The
% GraphBLAS prod (G,...) uses only a type of 'native', and a nanflag of
% 'includenan'.  See 'help prod' for more details.
%
% See also GrB/all, GrB/max, GrB/min, GrB/sum.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;
type = gbtype (G) ;
if (isequal (type, 'logical'))
    op = '&.logical' ;
else
    op = '*' ;
end

if (nargin == 1)
    C = GrB (gb_prod (op, type, G)) ;
else
    C = GrB (gb_prod (op, type, G, option)) ;
end

