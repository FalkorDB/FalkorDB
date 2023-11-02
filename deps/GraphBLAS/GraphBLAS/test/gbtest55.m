function gbtest55
%GBTEST55 test disp

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('GrB/display method with no semi-colon:\n') ;
H = GrB (rand (6)) %#ok<*NOPRT>

fprintf ('default:\n') ;
disp (H) ;
for level = 0:5
    disp (H, level) ;
end

fprintf ('gbtest55: all tests passed\n') ;

