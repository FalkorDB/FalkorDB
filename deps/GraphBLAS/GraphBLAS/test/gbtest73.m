function gbtest73 (ghb)
%GBTEST73 test GrB.normdiff

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

x = rand (5, 1) ;
y = rand (5, 1) ;
e1 = gtb_normdiff (ghb, x, y) ;
e2 = norm (x-y) ;
assert (abs (e1 - e2) < 1e-12) ;

try
    y = rand (2, 4) ;
    e1 = gtb_normdiff (ghb, x, y) ;
    ok = false ;
catch expected_error
    expected_error
    ok = true ;
end
assert (ok) ;

fprintf ('gbtest73 (%d): all tests passed\n', ghb) ;

