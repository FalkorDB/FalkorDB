function gbtest57 (ghb)
%GBTEST57 test fprintf and sprintf

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

c1 = fprintf ('pi: %g\n', pi) ;
c2 = fprintf ('pi: %g\n', gtb (ghb, pi)) ;
assert (c1 == c2) ;

s1 = sprintf ('pi: %g\n', pi) ;
s2 = sprintf ('pi: %g\n', gtb (ghb, pi)) ;
assert (isequal (s1, s2)) ;

A = int16 (magic (4)) ;
G = gtb (ghb, A) ;

c1 = fprintf ('%g\n', A) ;
c2 = fprintf ('%g\n', G) ;
assert (c1 == c2) ;

s1 = sprintf ('%g\n', A) ;
s2 = sprintf ('%g\n', G) ;
assert (isequal (s1, s2)) ;

A = speye (2) ;
G = gtb (ghb, A) ;

c1 = fprintf ('%g\n', full (A)) ;
c2 = fprintf ('%g\n', G) ;
assert (c1 == c2) ;

s1 = sprintf ('%g\n', full (A)) ;
s2 = sprintf ('%g\n', G) ;
assert (isequal (s1, s2)) ;

A = logical (A) ;
G = gtb (ghb, A) ;

c1 = fprintf ('%g\n', full (A)) ;
c2 = fprintf ('%g\n', G) ;
assert (c1 == c2) ;

fprintf ('gbtest57 (%d): all tests passed\n', ghb) ;

