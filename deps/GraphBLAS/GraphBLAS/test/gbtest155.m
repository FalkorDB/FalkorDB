function gbtest155
%GBTEST155 test GhB.get and GhB.set

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

B = sprand (3,3, 0.5) ;
s = GhB.get (B, 'format')
assert (isequal (s, 'sparse/hypersparse/bitmap/full by col')) ;

A = GhB (B) ;

s = GhB.get (A, 'format') ;
assert (isequal (s, 'sparse/hypersparse/bitmap/full by col')) ;

GhB.set (A, 'format', 'by row')
A
s = GhB.get (A, 'format') ;
assert (isequal (s, 'sparse/hypersparse/bitmap/full by row')) ;

GhB.set (A, 'format', 'sparse/hypersparse')
A
s = GhB.get (A, 'format') ;
assert (isequal (s, 'sparse/hypersparse by row')) ;

GhB.set (A, 'format', 'sparse/full')
A
s = GhB.get (A, 'format') ;
assert (isequal (s, 'sparse/full by row')) ;

GhB.set (A, 'format', 'sparse by col')
A
s = GhB.get (A, 'format') ;
assert (isequal (s, 'sparse by col')) ;

GhB.set (A, 'iso', 1)
A
s = GhB.get (A, 'iso') ;
assert (s == false) ;

GhB.set (A, 'iso', 0)
A
s = GhB.get (A, 'iso') ;
assert (s == false) ;

GhB.assign (A, A, pi) 
A
s = GhB.get (A, 'iso') ;
assert (s == false) ;

GhB.set (A, 'iso', 0)
A
s = GhB.get (A, 'iso') ;
assert (s == false) ;

GhB.set (A, 'iso', 1)
A
s = GhB.get (A, 'iso') ;
assert (s == true) ;

s = GhB.get (A, 'offset') ;
assert (s == 32) ;
s = GhB.get (A, 'row') ;
assert (s == 32) ;
s = GhB.get (A, 'col') ;
assert (s == 32) ;

GhB.set (A, 'offset', 64)
A
s = GhB.get (A, 'offset') ;
assert (s == 64) ;
s = GhB.get (A, 'row') ;
assert (s == 32) ;
s = GhB.get (A, 'col') ;
assert (s == 32) ;

GhB.set (A, 'row', 64)
A
s = GhB.get (A, 'offset') ;
assert (s == 64) ; 
s = GhB.get (A, 'row') ;
assert (s == 64) ;
s = GhB.get (A, 'col') ;
assert (s == 32) ;

GhB.set (A, 'col', 64)
A
s = GhB.get (A, 'offset') ;
assert (s == 64) ;
s = GhB.get (A, 'row') ;
assert (s == 64) ;
s = GhB.get (A, 'col') ;
assert (s == 64) ;

formats = {
    'hypersparse', 
    'sparse', 
    'sparse/hypersparse', 
    'bitmap', 
    'hypersparse/bitmap', 
    'sparse/bitmap', 
    'sparse/hypersparse/bitmap', 
    'full', 
    'hypersparse/full', 
    'sparse/full', 
    'sparse/hypersparse/full', 
    'bitmap/full', 
    'hypersparse/bitmap/full', 
    'sparse/bitmap/full', 
    'sparse/hypersparse/bitmap/full' } ;

for k = 1:length (formats)
    f = formats {k} ;
    GhB.set (A, 'format', f) ;
    s = GhB.get (A, 'format') ;
    assert (isequal (s, [f ' by col'])) ;
end

try
    GhB.set (A, 'garbage', 32) ;
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'invalid state')) ;

try
    GhB.set (A, 'row', 42) ;
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'invalid value (must be 32 or 64)')) ;

try
    GhB.set (B, 'offset', 32) ;
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'input matrix must be a GhB matrix')) ;

try
    s = GhB.get (A, 'garbage') ;
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'invalid state')) ;

fprintf ('gbtest155: all tests passed\n') ;

