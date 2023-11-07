function Y = dnn_builtin (W, bias, Y0)
%DNN_BUILTIN Sparse deep neural network without @GrB
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.
%
% Slightly revised from graphchallenge.org.
%
% Usage:
%
%   Y = dnn_builtin (W, bias, Y0)
%
% See also GrB.dnn, dnn_builtin2gb.

% note: this is now ported to Octave, by avoiding the use of singleton
% expansion.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Y = Y0 ;
n = size (Y, 2) ;

for i=1:length(W)
    % Propagate through layer.
    Z = Y * W {i} ;
    % Apply bias to non-zero entries.
    b = spdiags (bias{i}', 0, n, n) ;
    Y = Z + (double(logical(Z)) * b) ;
    % Threshold negative values.
    Y (Y < 0) = 0 ;
    % Threshold maximum values.
    Y (Y > 32) = 32 ;
end


