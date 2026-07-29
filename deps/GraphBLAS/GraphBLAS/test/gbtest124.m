function gbtest124 (ghb)
%GBTEST124 test [GrB,GhB].binops

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

list = gtb_binops (ghb) ;
gtb_binops (ghb) ;
have_octave = gb_octave ;
if (ghb && ~have_octave)
    help GhB.binops ;
else
    help GrB.binops ;
end

fprintf ('\ngbtest124 (%d): all tests passed\n', ghb) ;

