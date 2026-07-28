function C = reshape (G, varargin)
%RESHAPE reshape a matrix.
% C = reshape (G, m, n) or C = reshape (G, [m n]) returns the m-by-n matrix
% whose elements are taken columnwise from G.  The matrix G must have
% numel (G) == m*n.  That is numel (G) == numel (C) must be true.
%
% An optional parameter allows G to be to be reshaped row-wise, as:
% C = reshape (G, m, n, 'by row') or C = reshape (G, [m n], 'by row').  The
% default is 'by column'.
%
% See also GhB/numel, squeeze.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_reshape (1, G, varargin {:}) ;

