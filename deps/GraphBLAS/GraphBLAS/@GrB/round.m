function C = round (G)
%ROUND round entries of a matrix to the nearest integers.
% C = round (G) rounds the entries of G to the nearest integers.
%
% Note: the additional parameters of the built-in round function,
% round(x,n) and round (x,n,type), are not supported.
%
% See also GrB/ceil, GrB/floor, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: round (x,n) and round (x,n,type)

Q = G.opaque ;

if (gb_isfloat (gbtype (Q)) && gbnvals (Q) > 0)
    C = GrB (gbapply ('round', Q)) ;
else
    C = G ;
end

