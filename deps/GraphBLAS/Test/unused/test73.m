function test73
%TEST73 performance of C = A*B, with mask

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n----------------- C=A*B performance\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

Prob = ssget (1338)
A = Prob.A ;

    % remove the diagonal and extract L and U
    A = spones (A) ;
    A = spones (A+A') ;
    L = tril (A,-1) ;
    U = triu (A,1) ;
    A = L + U ;
    n = size (A,1) ;
    nz = nnz (L) ;

            semiring.multiply = 'times' ;
            semiring.add = 'plus' ;
            semiring.class = 'double' ;

Cin = sparse (n, n) ;
S = spones (L) ;

dnn = [ ] ;
dtn = struct ('inp0', 'tran') ;

% no mask ---------------------------------------------
tic
C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, L, L, dnn);
t1 = toc ;
fprintf ('mxm, no mask %g\n', t1) ;

tic
C1 = GB_mex_mxm  (Cin, [ ], [ ], semiring, L, L, dnn);
t1 = toc ;
fprintf ('mxm, no mask %g\n', t1) ;

tic
C2 = L*L ;
t2 = toc ;
fprintf ('built-in, no mask %g\n', t2) ;

% with mask ---------------------------------------------

tic
C2b = (L*L) .* L ;
t2 = toc ;
fprintf ('built-in, mask %g\n', t2) ;

tic
C3 = GB_mex_mxm  (Cin, L, [ ], semiring, L, L, dnn);
t3 = toc ;
fprintf ('mxm, with mask %g\n', t3) ;

tic
C5 = GB_mex_mxm  (Cin, [ ], [ ], semiring, L, L, dnn);
C5.matrix = C5.matrix .* L ;
t3 = toc ;
fprintf ('mxm, then emult %g\n', t3) ;

tic
C4 = GB_mex_mxm  (Cin, C2, [ ], semiring, L, L, dnn);
t4 = toc ;
fprintf ('mxm, with mask C %g\n', t4) ;

tic
C5 = GB_mex_mxm  (Cin, L, [ ], semiring, U, L, dtn);
t5 = toc ;
fprintf ('mxm, with mask L %g (dot)\n', t5) ;

assert (isequal (C2, C1.matrix)) ;
assert (isequal (C2 .* spones (L), C3.matrix)) ;
assert (isequal (C2, C4.matrix)) ;
assert (isequal (C2b, C5.matrix)) ;
nthreads_set (save, save_chunk) ;
