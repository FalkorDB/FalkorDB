function gbtest126 (ghb)
%GBTEST126 test GrB.selectops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

list = gtb_selectops (ghb)
gtb_selectops (ghb) ;
have_octave = gb_octave ;
if (ghb && ~have_octave)
    help GhB.selectops ;
else
    help GrB.selectops ;
end

fprintf ('\ngbtest126 (%d): all tests passed\n', ghb) ;

