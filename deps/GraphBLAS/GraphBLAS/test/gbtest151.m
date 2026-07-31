
function gbtest151
%GBTEST151 test error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

ok = true ;
C = GhB (rand (3)) ;

try
    C = GhB.apply (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.apply2 (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.apply2 (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.apply2 (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.assign (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.assign (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.assign (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.eadd (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.eadd (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.eadd (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.emult (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.emult (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.emult (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.eunion (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.eunion (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.eunion (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.kronecker (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.kronecker (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.kronecker (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.mxm (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.mxm (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.mxm (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.reduce (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.reduce (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.reduce (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.select (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.select (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.select (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.subassign (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.subassign (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.subassign (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.trans ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GrB.trans ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.trans (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GrB.vreduce (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.vreduce (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GhB.vreduce (C) ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    GrB.extract ;
    ok = false ;
catch expected_error
end
assert (ok) ;

try
    C = GhB.extract ;
    ok = false ;
catch expected_error
    msg = expected_error.message ;
end
assert (ok) ;
fprintf ('expected error: %s\n', msg) ;

fprintf ('gbtest151: all tests passed\n') ;

