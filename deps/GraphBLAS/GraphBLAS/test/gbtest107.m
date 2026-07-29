function gbtest107 (ghb)
%GBTEST107 test cell2mat error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

ok = true ;
try
    C = gtb_cell2mat (ghb, 'crud') ; %#ok<*NASGU>
    ok = false ;
catch me
    fprintf ('error expected: %s\n', me.message) ;
end
assert (ok) ;

try
    S = cell (2,2,2) ;
    C = gtb_cell2mat (ghb, S) ;
    ok = false ;
catch me
    fprintf ('error expected: %s\n', me.message) ;
end
assert (ok) ;

fprintf ('\ngbtest107 (%d): all tests passed\n', ghb) ;

