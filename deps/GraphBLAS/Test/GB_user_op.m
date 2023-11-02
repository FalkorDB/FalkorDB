function [z tol] = GB_user_op (op, x, y)
%GB_USER_OP apply a complex binary and unary operator
%
% built-in equivalents of the GraphBLAS user-defined Complex operators.
%
% [z tol] = GB_user_op (op,x,y) returns tol true if GB_mex_op(op,x,y) is
% allowed to have roundoff error when compared with GB_user_op(op,x,y).  tol is
% false if the result with built-in methods and GraphBLAS should match exactly.
%
% No typecasting is done for user-defined operators.  x,y,z are either
% double complex or double

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

tol = false ;

switch op

    %---------------------------------------------------------------------------
    % binary operators
    %---------------------------------------------------------------------------

    % x,y,z all complex:
    case 'first'
        z = x ;
    case 'second'
        z = y ;
    case { 'pair', 'oneb' }
        z = GB_spec_ones (size (x), GB_spec_type (x)) ;
%   case 'min'
%       z = min (x,y,'includenan') ;
%   case 'max'
%       z = max (x,y,'includenan') ;
    case 'plus'
        z = x+y ;
    case 'minus'
        z = x-y ;
    case 'rminus'
        z = y-x ;
    case 'times'
        z = x.*y ;
    case 'div'
        % built-in methods don't return a complex NaN (Nan+1i*Nan), but the
        % GraphBLAS GB_mex_op mexFunction does.  So if z has any of them,
        % replace them with a complex Nan, just to make sure the tests pass...
        z = x./y ;
        if (any (isnan (z)))
            z (isnan (z)) = complex (nan,nan) ;
        end
        tol = true ;
    case 'rdiv'
        z = y./x ;
        if (any (isnan (z)))
            z (isnan (z)) = complex (nan,nan) ;
        end
        tol = true ;
    case 'pow'
        z = y.^x ;
        tol = true ;

    % x,y,z all complex:
    case 'iseq'
        z = complex (double (x == y), 0) ;
    case 'isne'
        z = complex (double (x ~= y), 0) ;
    case 'isgt'
        z = complex (double (x >  y), 0) ;
    case 'islt'
        z = complex (double (x <  y), 0) ;
    case 'isge'
        z = complex (double (x >= y), 0) ;
    case 'isle'
        z = complex (double (x <= y), 0) ;

    % x,y,z all complex:
    case 'or'
        z = complex (double ((x ~= 0) | (y ~= 0)), 0) ;
    case 'and'
        z = complex (double ((x ~= 0) & (y ~= 0)), 0) ;
    case 'xor'
        z = complex (double ((x ~= 0) ~= (y ~= 0)), 0) ;

    % x,y complex, z logical:
    case 'eq'
        z = x == y ;
    case 'ne'
        z = x ~= y ;
    case 'gt'
        z = x >  y ;
    case 'lt'
        z = x <  y ;
    case 'ge'
        z = x >= y ;
    case 'le'
        z = x <= y ;

    % x,y double, z comlex:
    case 'complex'
        z = complex (x,y) ;

    %---------------------------------------------------------------------------
    % unary operators
    %---------------------------------------------------------------------------

    % x,z complex
    case 'one'
        z = GB_spec_ones (size (x), GB_spec_type (x))  ;
    case 'identity'
        z = x ;
    case 'ainv'
        z = -x ;
    case 'abs'
        z = abs (x) ;
        tol = true ;
    case 'minv'
        z = 1./x ;
        if (any (isnan (z)))
            z (isnan (z)) = complex (nan,nan) ;
        end
        tol = true ;
%   case 'not'
%       z = complex (double (~(x ~= 0)), 0) ;
    case 'conj'
        z = conj (x) ;

    % x complex, z real
    case 'real'
        z = real (x) ;
    case 'imag'
        z = imag (x) ;
    case { 'angle', 'carg' }
        z = angle (x) ;
        tol = true ;

%   case 'abs'
%       complex (abs (x), 0) ;
%   case 'cabs'
%       z = abs (x) ;
%       tol = true ;
%   case 'complex_real'
%       z = complex (x,0) ;
%   case 'complex_imag'
%       z = complex (0,x) ;

    otherwise
        error ('unrecognized complex operator')
end

