function gbtest52 (ghb)
%GBTEST52 test [GrB,GhB].format

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

gtb_format (ghb)
gtb_format (ghb, 'by col') ;
f = gtb_format (ghb) %#ok<*NOPRT>
assert (isequal (f, 'by col')) ;
A = magic (4)
G = gtb (ghb, A)
assert (isequal (f, gtb_format (ghb, G))) ;
gtb_format (ghb, 'by row')
f = gtb_format (ghb) %#ok<*NASGU>
assert (isequal (f, 'by row')) ;

H = gtb (ghb, 5,5)
assert (isequal ('by row', gtb_format (ghb, H))) ;

H = gtb (ghb, 5,5, 'by row')
assert (isequal ('by row', gtb_format (ghb, H))) ;

H = gtb (ghb, 5,5, 'by col')
assert (isequal ('by col', gtb_format (ghb, H))) ;

gtb_format (ghb, 'by col')
f = gtb_format (ghb)
assert (isequal (f, 'by col')) ;

H = gtb (ghb, 5,5)
assert (isequal ('by col', gtb_format (ghb, H))) ;

H = gtb (ghb, 5,5, 'by row')
assert (isequal ('by row', gtb_format (ghb, H))) ;

H = gtb (ghb, 5,5, 'by col')
assert (isequal ('by col', gtb_format (ghb, H))) ;

fprintf ('test GrB.format errors:\n') ;
ok = true ;
try
    [f, gunk] = gtb_format (ghb) ;
    ok = false ;
catch expected_error
    fprintf ('expected: %s\n', expected_error.message) ;
end
assert (ok) ;

fprintf ('gbtest52 (%d): all tests passed\n', ghb) ;

