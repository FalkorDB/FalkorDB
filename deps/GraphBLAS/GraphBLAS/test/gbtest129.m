function gbtest129 (ghb)
%GBTEST129 test [GrB,GhB].jit

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('gbtest129: testing %s.jit\n', gtb_name) ;

[status1, path1] = gtb_jit (ghb) ;
fprintf ('JIT: %s at %s\n', status1, path1) ;
gtb_jit (ghb, 'off', '/tmp') ;
[status2, path2] = gtb_jit (ghb) ;
assert (isequal (status2, 'off')) ;
assert (isequal (path2, '/tmp')) ;

[status3, path3] = gtb_jit (ghb, status1) ;
assert (isequal (status1, status3)) ;
assert (isequal (path2, path3)) ;

[status3, path3] = gtb_jit (ghb, status1, path1) ;
assert (isequal (status1, status3)) ;
assert (isequal (path1, path3)) ;

try
    gtb_jit (ghb, 0,0)
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'status must be a string')) ;

try
    gtb_jit (ghb, 'on',0)
    ok = false ;
catch me
    msg = me.message ;
    ok = true ;
end
assert (ok) ;
assert (gb_contains (msg, 'path must be a string')) ;

fprintf ('\njit status and path:\n') ;
gtb_jit (ghb) ;

fprintf ('\ngbtest129 (%d): all tests passed\n', ghb) ;

