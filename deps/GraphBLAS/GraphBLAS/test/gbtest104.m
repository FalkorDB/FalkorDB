function gbtest104 (ghb)
%GBTEST104 test formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

A = gtb (ghb, rand (4), 'sparse') %#ok<*NOPRT>
A = gtb (ghb, A, 'hypersparse')
A = gtb (ghb, A, 'bitmap')
A = gtb (ghb, A, 'full') %#ok<*NASGU>

fprintf ('\ngbtest104 (%d): all tests passed\n', ghb) ;

