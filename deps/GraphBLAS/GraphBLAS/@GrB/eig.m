function [V, varargout] = eig (G, varargin)
%EIG Eigenvalues and eigenvectors of a GraphBLAS matrix.
% See 'help eig' for details.
%
% See also eigs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% convert G to a built-in matrix
if (isreal (G) && issymmetric (G))
    % G can be sparse if G is real and symmetric
    A = double (G) ;
else
    % otherwise, G must be full.
    A = full (double (G)) ;
end

% use the built-in eig
if (nargin == 1)
    [V, varargout{1:nargout-1}] = builtin ('eig', A) ;
else
    for k = 1:length (varargin)
        argk = varargin {k} ;
        if (isobject (argk))
            varargin {k} = full (double (argk)) ;
        end
    end
    [V, varargout{1:nargout-1}] = builtin ('eig', A, varargin {:}) ;
end

