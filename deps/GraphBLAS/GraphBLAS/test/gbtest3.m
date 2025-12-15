function gbtest3
%GBTEST3 test dnn

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
levels = 4 ;
nfeatures = 6 ;
nneurons = 16 ;

for level = 1:levels
    W {level} = sprand (nneurons, nneurons, 0.5) ; %#ok<*AGROW>
    bias {level} = -0.3 * ones (1, nneurons) ;
end

Y0 = sprandn (nfeatures, nneurons, 0.5) ;

tic
Y1 = dnn_builtin (W, bias, Y0) ;
toc

[W, bias, Y0] = dnn_builtin2gb (W, bias, Y0) ;
tic
Y2 = GrB.dnn (W, bias, Y0) ;
toc

err = norm (Y1-Y2,1) ;
assert (err < 1e-5) ;

help GrB.dnn
fprintf ('gbtest3: all tests passed\n') ;

