function Y = dnn (W, Bias, Y0)
%GHB.DNN Sparse deep neural network in GraphBLAS.
% Performs ReLU inference using input feature vector(s) Y0, DNN weights W, and
% diagonal Bias matrices.  The input features are in a matrix Y0 of size
% nfeatures- by-nneurons.  The DNN weights W is a cell array with W{k} being
% the kth layer of the DNN, so that the number of layers is nlayers = length
% (W).  W{k} is a matrix of size nneurons-by-nneurons.  The Bias variable is a
% cell array of length nlayers.  Each Bias{k} is a diagonal matrix of size
% nneurons- by-nneurons, which gives the Bias values of each neuron in the kth
% layer.
%
% This method solves the Sparse Deep Neural Network Graph Challenge; see
% https://graphchallenge.mit.edu/challenges/ for details.
%
% Usage:
%
%   Y = GhB.dnn (W, Bias, Y0) ;             % using GraphBLAS
%   Y = inferenceReLUvec (W, bias, Y0) ;    % MIT reference implementation
%
% The matrices can be stored by row or by column, but GhB.format ('by row') is
% somewhat faster.  For the MIT GraphChallenge, all matrices can be 'single',
% and the same results are obtained.
%
% In the original reference implementation, the bias{k} is a row vector of size
% 1-by-nneurons.  The reference inputs can be converted to GraphBLAS matrices
% with the following code:
%
%   desc = struct ('format', 'by row') ;
%   n = size (Y0, 2) ;
%   Y0 = GhB (Y0, 'single', 'by row') ;
%   for k=1:length(W)
%       W {k} = GhB (W {k}, 'single', 'by row') ;
%       Bias {k} = GhB.build (1:n, 1:n, bias {k}, n, n, '+', 'single', desc) ;
%   end
%
% All of the above conversion is optional, except for Bias {k} since it is
% changed from a row vector to a diagonal matrix.
%
% See also GraphBLAS/demo/dnn_builtin, GraphBLAS/demo/dnn_builtin2gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (GhB.isbyrow (Y0))
    % hypersparse-by-row is fastest, since entire rows drop out of Y
    desc.format = 'hyper by row' ;
    Y = GhB (Y0, 'hyper by row') ;
else
    desc.format = 'by col' ;
    Y = GhB (Y0) ;
end
ymax = single (32) ;

for k = 1:length(W)
    % Propagate through layer, apply bias, and threshold negative values.
    GhB.mxm (Y, Y, '+.*', W {k}, desc) ;    % Y = Y * W {k}
    GhB.mxm (Y, Y, '+.+', Bias {k}, desc)   % apply bias to entries in Y
    GhB.select (Y, Y, '>0', desc) ;         % only keep entries >0 in Y
    M = Y > ymax ;                          % find entries over threshold in Y
    if (nnz (M) > 0)
        GhB.subassign (Y, M, ymax) ;        % Y (M) = ymax ;
    end
end

