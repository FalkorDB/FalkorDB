function gbtest131 (ghb)
%GBTEST131 misc error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = gtb (ghb, magic (4)) ;

try
    d = diag (A, [1 2]) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'k must be a scalar')) ;

try
    C = gtb_apply2 (ghb, A, '*', [1 2]) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'either A or B must be a non-empty scalar')) ;

try
    C = gtb_apply2 (ghb, A, '*', sparse (0)) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'either A or B must be a non-empty scalar')) ;

try
    result = isa (GrB (pi), pi) ;
    ok = false ;
    msg = '' ;
catch expected_error
    ok = true ;
    msg = expected_error.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'type must be a string')) ;

fprintf ('\ngbtest131 (%d): all tests passed\n', ghb) ;

