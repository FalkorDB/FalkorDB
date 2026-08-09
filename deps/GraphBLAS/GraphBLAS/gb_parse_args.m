function [m, n, type] = gb_parse_args (func, varargin)
%GB_PARSE_ARGS parse arguments for various functions.  Not user-callable.
% For true, false, ones, zeros, eye, and speye.  For example:
%
%   C = GrB.ones ;
%   C = GrB.ones (n) ;
%   C = GrB.ones (m,n) ;
%   C = GrB.ones ([m n]) ;
%   C = GrB.ones (... , 'like', G) ;
%   C = GrB.ones (... , 'int8') ;

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% parse the type
type = 'double' ;
nargs = length (varargin) ;
for k = 1:nargs
    arg = varargin {k} ;
    if (ischar (arg))
        if (isequal (arg, 'like'))
            if (nargs ~= k+1)
                error ('GrB:error', 'usage: GrB.%s (m, n, ''like'', G)', func) ;
            end
            type = gb_type (varargin {k+1}) ;
        else
            if (nargs ~= k)
                error ('GrB:error', 'usage: GrB.%s (m, n, type)', func) ;
            end
            type = arg ;
        end
        nargs = k-1 ;
        break ;
    end
end

% parse the dimensions
[m, n] = gb_parse_dimensions (varargin {1:nargs}) ;

