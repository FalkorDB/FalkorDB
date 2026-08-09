
function gbtest152 (ghb)
%GBTEST152 test [GrB,GhB].nvals

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;

A = Problem.A ;
G = gtb (ghb, A) ;
 
anvals = gtb_nvals (ghb, A) ;
cnvals = gtb_nvals (ghb, G) ;

assert (isequal (cnvals, anvals)) ;
assert (isequal (cnvals, 1888)) ;

fprintf ('gbtest152 (%d): all tests passed\n', ghb) ;

