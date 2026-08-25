function gbtest156
%GBTEST156 test GhB.bfs error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = Problem.A ;
deg = GhB.entries (A, 'row', 'degree') ;
AT = logical (spones (A))' ;
n = size (A, 1) ;

% error handling
try
    v = GhB.bfs (A, AT, 1) ;
    ok = false ;
catch me
    ok = true ;
    msg = me.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'usage:')) ;

try
    v = GhB.bfs (A, rand (2)) ;
    ok = false ;
catch me
    ok = true ;
    msg = me.message ;
end
assert (ok) ;
assert (gb_contains (msg, 's must be a scalar')) ;

try
    v = GhB.bfs (A, rand (2), deg, 1) ;
    ok = false ;
catch me
    ok = true ;
    msg = me.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'AT has the wrong size')) ;

try
    v = GhB.bfs (A, AT, deg, 1, 'undirected', 'check') ;
    ok = false ;
catch me
    ok = true ;
    msg = me.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'A must be symmetric')) ;

try
    v = GhB.bfs (A, A, deg, 1, 'check') ;
    ok = false ;
catch me
    ok = true ;
    msg = me.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'spones(A) must equal spones(AT)''')) ;

crud = ones (n, 1) ;
try
    v = GhB.bfs (A, AT, crud, 1, 'check') ;
    ok = false ;
catch me
    ok = true ;
    msg = me.message ;
end
assert (ok) ;
assert (gb_contains (msg, 'degree is incorrect')) ;

fprintf ('gbtest156: all tests passed\n') ;

