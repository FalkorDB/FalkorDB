function [xdomain, ydomain] = GB_spec_opdomain (op)
%GB_SPEC_OPDOMAIN determine domain of a unary or binary operator
%
%   [xdomain, ydomain] = GB_spec_opdomain (op) determines the domains of
%   x and y where the unary or binary operator is valid.  Each output is an
%   array of size two.  Divide-by-zero is ignored.
%
%   An error results if the operator is not defined at all (op.opname = 'erf'
%   and op.opname = 'single complex' for example).
%
% See also GB_spec_op.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the operator name and type
[opname optype ztype xtype ytype] = GB_spec_operator (op, 'double') ;

xdomain = [-inf, inf] ;
ydomain = [-inf, inf] ;

if (test_contains (optype, 'complex'))
    % complex operators z=f(x,y) are valid over all x and y
    return
end

% find the domain of a real operator

switch opname

    case { 'pow', 'sqrt', 'log', 'log10', 'log2', 'gammaln', 'lgamma' }
        xdomain = [0, inf] ;

    case { 'asin', 'acos', 'atanh' }
        xdomain = [-1, 1] ;

    case { 'acosh', 'asech' }
        xdomain = [1, inf] ;

    case 'log1p'
        xdomain = [-1, inf] ;

    otherwise
        % op is defined for all x and y

end

