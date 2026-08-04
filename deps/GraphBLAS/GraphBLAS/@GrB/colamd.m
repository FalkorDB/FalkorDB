function [p, varargout] = colamd (G, opts)
%COLAMD column approximate minimum degree ordering.
% See 'help colamd' for details.
%
% See also GrB/amd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    [p, varargout{1:nargout-1}] = colamd (logical (G)) ;
else
    [p, varargout{1:nargout-1}] = colamd (logical (G), double (opts)) ;
end

