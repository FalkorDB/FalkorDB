function C = sign (G)
%SIGN signum function.
% C = sign (G) is the signum function for each entry of G.  For real
% values, sign(x) is 1 if x > 0, zero if x is zero, and -1 if x < 0.
% For the complex case, sign(x) = x ./ abs (x).
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Q = G.opaque ;
type = gbtype (Q) ;

if (isequal (type, 'logical'))
    C = G ;
elseif (~gb_isfloat (type))
    C = GrB (gbnew (gbapply ('signum.single', Q), type)) ;
else
    C = GrB (gbapply ('signum', Q)) ;
end

