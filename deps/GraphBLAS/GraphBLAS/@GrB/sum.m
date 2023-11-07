function C = sum (G, option)
%SUM sum of elements.
% C = sum (G), where G is an m-by-n matrix, computes a 1-by-n row vector C
% where C(j) is the sum of all entries in G(:,j).  If G is a row or column
% vector, then sum (G) is a scalar sum of all the entries in the vector.
%
% C = sum (G,'all') sums all elements of G to a single scalar.
%
% C = sum (G,1) is the default when G is a matrix, which is to sum each
% column to a scalar, giving a 1-by-n row vector.  If G is already a row
% vector, then C = G.
%
% C = sum (G,2) sums each row to a scalar, resulting in an m-by-1 column
% vector C where C(i) is the sum of all entries in G(i,:).
%
% The built-in sum (A, ... type, nanflag) allows for different types of
% sums to be computed, and the NaN behavior can be specified.  The
% GraphBLAS sum (G,...) uses only a type of 'native', and a nanflag of
% 'includenan'.  See 'help sum' for more details.
%
% See also GrB/any, GrB/prod, GrB/max, GrB/min.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

G = G.opaque ;

if (isequal (gbtype (G), 'logical'))
    op = '+.int64' ;
else
    op = '+' ;
end

if (nargin == 1)
    C = GrB (gb_sum (op, G)) ;
else
    C = GrB (gb_sum (op, G, option)) ;
end

