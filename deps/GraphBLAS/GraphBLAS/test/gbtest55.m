function gbtest55 (ghb)
%GBTEST55 test disp and [GrB,GhB].print

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('%s/display method with no semi-colon:\n', gtb_name) ;
H = gtb (ghb, rand (6)) %#ok<*NOPRT>

fprintf ('default:\n') ;
disp (H) ;
for level = 0:5
    disp (H, level) ;
end

fprintf ('using %s.print, default:\n', gtb_name) ;
gtb_print (ghb, H) ;
for level = 0:5
    gtb_print (ghb, H, level) ;
end

fprintf ('using %s.print, for builtin:\n', gtb_name) ;
H = double (H)
gtb_print (ghb, H) ;
for level = 0:5
    gtb_print (ghb, H, level) ;
end

fprintf ('gbtest55 (%d): all tests passed\n', ghb) ;

