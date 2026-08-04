function C = gb_reshape (ghb, G, varargin)
% GB_RESHAPE implements reshape for GrB and GhB.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
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

C = gzb_reshape (ghb, G, mnew, nnew, by_col) ;

