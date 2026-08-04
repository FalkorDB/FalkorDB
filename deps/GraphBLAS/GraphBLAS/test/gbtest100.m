function gbtest100 (ghb)
%GBTEST100 test [GrB,GhB].ver and [GrB,GhB].version

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

if (ghb)
    GhB.MATLAB_vs_GrB ;
else
    GrB.MATLAB_vs_GrB ;
end

fprintf ('v = %s.ver\n', gtb_name) ;
v = gtb_ver (ghb) ;
display (v) ;

fprintf ('v = %s.version\n', gtb_name) ;
v = gtb_version (ghb) ;
display (v) ;

fprintf ('%s.ver\n\n', gtb_name) ;
gtb_ver (ghb)

