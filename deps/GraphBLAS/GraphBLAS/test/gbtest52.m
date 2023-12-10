function gbtest52
%GBTEST52 test GrB.format

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GrB.format
GrB.format ('by col') ;
f = GrB.format %#ok<*NOPRT>
assert (isequal (f, 'by col')) ;
A = magic (4)
G = GrB (A)
assert (isequal (f, GrB.format (G))) ;
GrB.format ('by row')
f = GrB.format %#ok<*NASGU>
assert (isequal (f, 'by row')) ;

H = GrB (5,5)
assert (isequal ('by row', GrB.format (H))) ;

H = GrB (5,5, 'by row')
assert (isequal ('by row', GrB.format (H))) ;

H = GrB (5,5, 'by col')
assert (isequal ('by col', GrB.format (H))) ;

GrB.format ('by col')
f = GrB.format
assert (isequal (f, 'by col')) ;

H = GrB (5,5)
assert (isequal ('by col', GrB.format (H))) ;

H = GrB (5,5, 'by row')
assert (isequal ('by row', GrB.format (H))) ;

H = GrB (5,5, 'by col')
assert (isequal ('by col', GrB.format (H))) ;

fprintf ('gbtest52: all tests passed\n') ;

