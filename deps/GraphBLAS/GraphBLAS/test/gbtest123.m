function gbtest123 (ghb)
%GBTEST123 test build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

n = 1000 ;
H = gtb (ghb, n, n) ;
H (1,1) = 1 ;
S = gtb_build (ghb, H,H,pi) ;
P = sparse (pi) ;
assert (isequal (S, P)) ;

n = flintmax ;
H = gtb (ghb, n, n) ;
H (1,1) = 1 ;
try
    S = gtb_build (ghb, H,H,H) ;
    ok = false ;
catch expected_error
    msg = expected_error.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'input matrix dimensions are too large')) ;

fprintf ('\ngbtest123 (%d): all tests passed\n', ghb) ;

