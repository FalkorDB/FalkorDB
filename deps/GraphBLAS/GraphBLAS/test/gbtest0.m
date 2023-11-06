function gbtest0
%GBTEST0 test GrB.clear

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GrB.clear
GrB.init

assert (isequal (GrB.format, 'by col')) ;
assert (isequal (GrB.chunk, 64*1024)) ;

GrB.burble (1) ;
GrB.burble (0) ;
assert (~GrB.burble) ;

GrB.burble (false) ;
assert (~GrB.burble) ;

ok = true ;
try
    GrB.burble (rand (2)) ;
    ok = false ;
catch me
    fprintf ('expected error:\n') ;
    disp (me) ;
end
assert (ok) ;

fprintf ('default # of threads: %d\n', GrB.threads) ;

fprintf ('gbtest0: all tests passed\n') ;

