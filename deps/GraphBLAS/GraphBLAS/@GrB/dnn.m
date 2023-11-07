function Y = dnn (W, bias, Y0)
%GRB.DNN Sparse deep neural network in GraphBLAS.
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W,
% and bias vectors.  The input features are in a matrix Y0 of size
% nfeatures-by- nneurons.  The DNN weights W is a cell array with W{k}
% being the kth layer of the DNN, so that the number of layers is nlayers =
% length (W).  W{k} is a matrix of size nneurons-by-nneurons.  The bias
% variable is a cell array of length nlayers.  Each bias{k} is a diagonal
% matrix of size nneurons-by-nneurons, which gives the bias values of each
% neuron in the kth layer.
%
% Usage:
%
%   Y = GrB.dnn (W, bias, Y0) ;
%
% The matrices can be stored by row or by column, but GrB.format ('by row')
% is somewhat faster.  For the 2019 GraphChallenge, all matrices can be
% 'single', and the same results are obtained.
%
% In the original reference implementation, the bias{k} is a row vector of
% size 1-by-nneurons.  The reference inputs can be converted to GraphBLAS
% matrices with the following code:
%
%   d = struct ('format', 'by row') ;
%   n = size (Y0, 2) ;
%   Y0 = GrB (Y0, 'single', 'by row') ;
%   for k=1:length(W)
%       W {k} = GrB (W {k}, 'single', 'by row') ;
%       bias {k} = GrB.build (1:n, 1:n, bias {k}, n, n, '+', 'single', d) ;
%   end
%
% All of the above conversion is optional, except for bias {k} since it is
% changed from a row vector to a diagonal matrix.
%
% See also dnn_builtin, dnn_builtin2gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: this is a high-level algorithm that uses GrB objects.

[f,~] = GrB.format (Y0) ;
desc.format = '' ;
if (isequal (f, 'by row'))
    % hypersparse-by-row is fastest, since entire rows drop out of Y
    desc.format = 'hyper by row' ;
end
tol = single (32) ;

Y = Y0 ;
for k = 1:length(W)
    % Propagate through layer, apply bias, and threshold negative values.
    Y = GrB.mxm (Y, '+.*', W {k}, desc) ;
    Y = GrB.select (GrB.mxm (Y, '+.+', bias {k}, desc), '>0', desc) ;
    M = Y > tol ;
    if (nnz (M) > 0)
        % Y (M) = tol ;
        Y = GrB.subassign (Y, M, tol, desc) ;
    end
end

