function Y = dnn (W, Bias, Y0)
%GRB.DNN Sparse deep neural network in GraphBLAS.
%
% See 'help GhB.dnn' for details.
% This method is identical, except that it returns GrB matrices.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Y = GrB (GhB.dnn (W, Bias, Y0)) ;

