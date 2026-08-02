function gbtest0 (ghb)
%GBTEST0 test GrB.clear and GhB.clear

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

gtb_clear (ghb) ;
gtb_init (ghb) ;

assert (isequal (gtb_format (ghb), 'by col')) ;
assert (isequal (gtb_chunk (ghb), 64*1024)) ;

gtb_burble (ghb, 1) ;
gtb_burble (ghb, 0) ;
assert (~gtb_burble (ghb)) ;

gtb_burble (ghb, false) ;
assert (~gtb_burble (ghb)) ;

ok = true ;
try
    gtb_burble (ghb, rand (2)) ;
    ok = false ;
catch me
    fprintf ('expected error:\n') ;
    disp (me) ;
end
assert (ok) ;

fprintf ('default # of threads: %d\n', gtb_threads (ghb)) ;

fprintf ('gbtest0 (%d): all tests passed\n', ghb) ;

