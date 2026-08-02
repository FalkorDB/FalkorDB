function gbtest3 (ghb)
%GBTEST3 test GrB.dnn and GhB.dnn

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

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

[W, bias, Y0] = dnn_builtin2gb (ghb, W, bias, Y0) ;
tic
Y2 = gtb_dnn (ghb, W, bias, Y0) ;
toc

err = norm (Y1-Y2,1) ;
assert (err < 1e-5) ;

% test again with all matrices held by colum

[W, bias, Y0] = dnn_builtin2gb (ghb, W, bias, Y0) ;
for level = 1:levels
    W {level} = gtb (ghb, W {level}, 'by col') ;
    bias {level} = gtb (ghb, bias {level}, 'by col') ;
end
Y0 = gtb (ghb, Y0, 'by col') ;

tic
Y2 = gtb_dnn (ghb, W, bias, Y0) ;
toc

err = norm (Y1-Y2,1) ;
assert (err < 1e-5) ;

if (ghb == 1)
    help GhB.dnn
else
    help GrB.dnn
end
fprintf ('gbtest3 (%d): all tests passed\n', ghb) ;

