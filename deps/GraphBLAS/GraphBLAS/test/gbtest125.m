function gbtest125 (ghb)
%GBTEST125 test [GrB,GhB].monoids

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

list = gtb_monoids (ghb)
gtb_monoids (ghb) ;
have_octave = gb_octave ;
if (ghb && ~have_octave)
    help GhB.monoids ;
else
    help GrB.monoids ;
end

fprintf ('\ngbtest125 (%d): all tests passed\n', ghb) ;
