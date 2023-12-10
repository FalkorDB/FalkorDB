function test160
%TEST160 test GrB_mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

n = 100 ;
Mask.matrix = (rand (n) > 0.5) ;
Mask.pattern = true (n) ;
mtypes = { 'int8', 'int16', 'int32', 'int64', 'double complex' } ;

Mask2 = GB_spec_random (n, n, 1, 0.01, 'logical') ;
Mask2.matrix = logical (Mask2.matrix) ;
Mask2.matrix (:,1) = false ;
Mask2.pattern (:,1) = false ;

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

dnn = struct ;
dnn_struct = struct ('mask', 'structural') ;
dnn_notM_struct = struct ('mask', 'structural complement') ;
dnn_notM = struct ('mask', 'complement') ;
dnn_notM_hash = struct ('mask', 'complement', 'axb', 'hash') ;
dnn_hash = struct ('axb', 'hash') ;

d = 0.01 ;

A = GB_spec_random (n, n, d, 1, 'double') ;
G = A ;
G.matrix (:,1:2) = 1 ;
G.pattern (:,1:2) = true ;
B = GB_spec_random (n, n, d, 1, 'double') ;
B.matrix (1:2,1) = 1 ;
B.pattern (1:2,1) = true ;
b = GB_spec_random (n, 1, d, 1, 'double') ;
Cin = sparse (n, n) ;
cin = sparse (n, 1) ;
mask.matrix = (rand (n,1) > 0.5) ;
mask.pattern = true (n,1) ;

H.matrix = sparse (ones (n,n)) ;
H.matrix (1,1) = 0 ;
H.pattern = sparse (true (n,n)) ;
H.matrix (1,1) = false ;
H.sparsity = 2 ;
mask2.matrix = sparse (false (n,1)) ;
mask2.matrix (1,1) = true ;
mask2.pattern = sparse (false (n,1)) ;
mask2.pattern (1,1) = true ;
x = GB_spec_random (n, 1, 0.5, 1, 'double') ;
x.sparsity = 2 ;
y = GB_spec_random (n, 1, 0.02, 1, 'double') ;
y.sparsity = 2 ;

K = GB_spec_random (1000, 2, 0.1, 1, 'double') ;
K.matrix (1:2, 1:2) = pi ;
K.pattern (1:2, 1:2) = true ;
K.sparsity = 2 ;
z.matrix = rand (2,1) ;
maskz.matrix = sparse (false (1000,1)) ;
maskz.matrix (1,1) = true ;
maskz.pattern = sparse (false (1000,1)) ;
maskz.pattern (1,1) = true ;
maskz.class = 'logical' ;
cinz = sparse (1000, 1) ;


for k = 1:length (mtypes)

    fprintf ('%s ', mtypes {k}) ;
    Mask.class = mtypes {k} ;
    Mask2.class = mtypes {k} ;
    mask.class = mtypes {k} ;

    % C<M> = A*B
    C1 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dnn) ;
    C2 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dnn) ;
    GB_spec_compare (C1, C2) ;

    % C<M,struct> = A*B
    C1 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dnn_struct) ;
    C2 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dnn_struct) ;
    GB_spec_compare (C1, C2) ;

    % C<!M,struct> = A*B
    C1 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dnn_notM_struct) ;
    C2 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dnn_notM_struct) ;
    GB_spec_compare (C1, C2) ;

    % C<!M> = A*B
    C1 = GB_spec_mxm (Cin, Mask, [ ], semiring, A, B, dnn_notM) ;
    C2 = GB_mex_mxm  (Cin, Mask, [ ], semiring, A, B, dnn_notM) ;
    GB_spec_compare (C1, C2) ;

    % C<M> = G*b
    C1 = GB_spec_mxm (cin, mask, [ ], semiring, G, b, dnn) ;
    C2 = GB_mex_mxm  (cin, mask, [ ], semiring, G, b, dnn) ;
    GB_spec_compare (C1, C2) ;

    % C<!M> = G*b
    C1 = GB_spec_mxm (cin, mask, [ ], semiring, G, b, dnn_notM) ;
    C2 = GB_mex_mxm  (cin, mask, [ ], semiring, G, b, dnn_notM) ;
    GB_spec_compare (C1, C2) ;

    % C<!M,struct> = A*B
    C1 = GB_spec_mxm (cin, mask, [ ], semiring, G, b, dnn_notM_struct) ;
    C2 = GB_mex_mxm  (cin, mask, [ ], semiring, G, b, dnn_notM_struct) ;
    GB_spec_compare (C1, C2) ;

    % C<!M> = A*B
    C1 = GB_spec_mxm (Cin, Mask2, [ ], semiring, A, B, dnn_notM) ;
    C2 = GB_mex_mxm  (Cin, Mask2, [ ], semiring, A, B, dnn_notM) ;
    GB_spec_compare (C1, C2) ;

    % C<!Mask2> = G*B
    C1 = GB_spec_mxm (Cin, Mask2, [ ], semiring, G, B, dnn_notM) ;
    C2 = GB_mex_mxm  (Cin, Mask2, [ ], semiring, G, B, dnn_notM) ;
    GB_spec_compare (C1, C2) ;

    % C<!M> = H*x
    C1 = GB_spec_mxm (cin, mask2, [ ], semiring, H, x, dnn_notM) ;
    C2 = GB_mex_mxm  (cin, mask2, [ ], semiring, H, x, dnn_notM) ;
    GB_spec_compare (C1, C2) ;

    % C<!M> = G*x
    C1 = GB_spec_mxm (cin, mask2, [ ], semiring, G, x, dnn_notM_hash) ;
    C2 = GB_mex_mxm  (cin, mask2, [ ], semiring, G, x, dnn_notM_hash) ;
    GB_spec_compare (C1, C2) ;

    % C<!M> = K*z
    z.sparsity = 4 ;
    C1 = GB_spec_mxm (cinz, maskz, [ ], semiring, K, z, dnn_notM_hash) ;
    C2 = GB_mex_mxm  (cinz, maskz, [ ], semiring, K, z, dnn_notM_hash) ;
    GB_spec_compare (C1, C2) ;

end

% C = K*z
z.sparsity = 2 ;
C1 = GB_spec_mxm (cinz, [ ], [ ], semiring, K, z, dnn_hash) ;
C2 = GB_mex_mxm  (cinz, [ ], [ ], semiring, K, z, dnn_hash) ;
C3 = GB_mex_mxm_generic  (cinz, [ ], [ ], semiring, K, z, dnn_hash) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (C1, C3) ;

% C<!M> = K*z
z.sparsity = 2 ;
C1 = GB_spec_mxm (cinz, maskz, [ ], semiring, K, z, dnn_notM_hash) ;
C2 = GB_mex_mxm  (cinz, maskz, [ ], semiring, K, z, dnn_notM_hash) ;
C3 = GB_mex_mxm_generic  (cinz, maskz, [ ], semiring, K, z, dnn_notM_hash) ;
GB_spec_compare (C1, C2) ;
GB_spec_compare (C1, C3) ;

fprintf ('\ntest160: all tests passed\n') ;
