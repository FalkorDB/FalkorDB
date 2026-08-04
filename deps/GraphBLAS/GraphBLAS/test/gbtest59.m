function gbtest59 (ghb)
%GBTEST59 test end

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = rand (4,7) ;
G = gtb (ghb, A) ;

A = A (2:end, 3:end) ;
G = G (2:end, 3:end) ;
assert (isequal (G, A)) ;

A = A (2:2:end, 3:2:end) ;
G = G (2:2:end, 3:2:end) ;
assert (isequal (G, A)) ;

A = rand (7, 1) ;
G = gtb (ghb, A) ;

A = A (2:2:end) ;
G = G (2:2:end) ;
assert (isequal (G, A)) ;

fprintf ('gbtest59 (%d): all tests passed\n', ghb) ;

