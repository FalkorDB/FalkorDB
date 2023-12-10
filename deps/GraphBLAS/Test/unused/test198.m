function test198
%TEST198 test apply with C=op(C) 

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

% GB_mex_burble (1) ;
n = 10 ;
C = sprand (n, n, 0.5) ;
op.opname = 'sqrt' ;
op.optype = 'double' ;

C1 = GB_mex_apply_alias (C, op) ;
C2 = sqrt (C) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

Ciso.matrix = pi * spones (C) ;
Ciso.iso = true ;

C1 = GB_mex_apply_alias (Ciso, op) ;
C2 = sqrt (Ciso.matrix) ;
err = norm (C1.matrix - C2, 1) ;
assert (err < 1e-12) ;

GB_mex_burble (0) ;
fprintf ('test198: all tests passed\n') ;

