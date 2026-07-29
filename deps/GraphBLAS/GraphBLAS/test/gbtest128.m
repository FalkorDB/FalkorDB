function gbtest128 (ghb)
%GBTEST128 test [GrB,GhB].unops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

list = gtb_unops (ghb)
gtb_unops (ghb) ;
have_octave = gb_octave ;
if (ghb && ~have_octave)
    help GhB.unops ;
else
    help GrB.unops ;
end

fprintf ('\ngbtest128 (%d): all tests passed\n', ghb) ;

