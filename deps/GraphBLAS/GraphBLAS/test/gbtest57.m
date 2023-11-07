function gbtest57
%GBTEST57 test fprintf and sprintf

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

c1 = fprintf ('pi: %g\n', pi) ;
c2 = fprintf ('pi: %g\n', GrB (pi)) ;
assert (c1 == c2) ;

s1 = sprintf ('pi: %g\n', pi) ;
s2 = sprintf ('pi: %g\n', GrB (pi)) ;
assert (isequal (s1, s2)) ;

A = int16 (magic (4)) ;
G = GrB (A) ;

c1 = fprintf ('%g\n', A) ;
c2 = fprintf ('%g\n', G) ;
assert (c1 == c2) ;

s1 = sprintf ('%g\n', A) ;
s2 = sprintf ('%g\n', G) ;
assert (isequal (s1, s2)) ;

A = speye (2) ;
G = GrB (A) ;

c1 = fprintf ('%g\n', full (A)) ;
c2 = fprintf ('%g\n', G) ;
assert (c1 == c2) ;

s1 = sprintf ('%g\n', full (A)) ;
s2 = sprintf ('%g\n', G) ;
assert (isequal (s1, s2)) ;

A = logical (A) ;
G = GrB (A) ;

c1 = fprintf ('%g\n', full (A)) ;
c2 = fprintf ('%g\n', G) ;
assert (c1 == c2) ;

fprintf ('gbtest57: all tests passed\n') ;

