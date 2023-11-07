function C = reshape (G, varargin)
%RESHAPE reshape a matrix.
% C = reshape (G, m, n) or C = reshape (G, [m n]) returns the m-by-n
% matrix whose elements are taken columnwise from G.  The matrix G must
% have numel (G) == m*n.  That is numel (G) == numel (C) must be true.
%
% An optional parameter allows G to be to be reshaped row-wise instead
% of columnwise:  C = reshape (G, m, n, 'by row') or C = 
% reshape (G, [m n], 'by row').  The default is 'by column'.
%
% See also GrB/numel, squeeze.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isobject (G))
    G = G.opaque ;
end

% the third output of gb_parse_args is not actually a type, but 'by row', 'by
% col', or 'double' if not present on input.
[mnew, nnew, type] = gb_parse_args ('reshape', varargin {:}) ;
mnew = int64 (mnew) ;
nnew = int64 (nnew) ;

switch (type)
    case 'by row'
        by_col = false ;
    case { 'by column', 'double' }
        % if type is 'double', the row/colwise parameter is not present
        by_col = true ;
    otherwise
        error ('GrB:error', 'unknown reshape option') ;
end

C = GrB (gbreshape (G, mnew, nnew, by_col)) ;

