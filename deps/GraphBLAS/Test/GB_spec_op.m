function C = GB_spec_op (op, A, B)
%GB_SPEC_OP apply a unary or binary operator
%
% Apply a binary operator z = f (x,y) element-wise to x and y, or a unary
% operator z = f(x) just x.  The operator op is any built-in GraphBLAS
% operator.
%
% op or op.opname is a string with just the operator name.  Valid names of
% binary operators are 'first', 'second', 'min', 'max', 'plus', 'minus',
% 'rminus', 'times', 'div', 'rdiv', 'eq', 'ne', 'gt', 'lt', 'ge', 'le', 'or',
% 'and', 'xor'.  'iseq', 'isne', 'isgt', 'islt', 'isge', 'le', 'pair' (same
% as 'oneb'), 'any', 'pow', ('bitget' or 'bget'), ('bitset' or 'bset'),
% ('bitclr' or 'bclr'), ('bitand' or 'band'), ('bitor' or 'bor'), ('bitxor' or
% 'bxor'), ('bitxnor', 'bxnor'), ('bitshift' or 'bshift'), ('bitnot' or
% 'bitcmp'), 'atan2', 'hypot', ('ldexp' or 'pow2'), ('complex', 'cmplx').  
%
% Unary operators are 'one', 'identity', 'ainv', 'abs', 'minv', 'not', 'bnot',
% 'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh',
% 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'signum', 'ceil', 'floor',
% 'round', ('trunc' or 'fix'), 'exp2', 'expm1', 'log10', 'log2', ('lgamma' or
% 'gammaln'), ('tgamma' or 'gamma'), 'erf', 'erfc', 'frexpx', 'frexpe', 'conj',
% ('creal' or 'real'), ('cimag' or 'imag'), ('carg' or 'angle'), 'isinf',
% 'isnan', 'isfinite', 'cbrt'.
%
% op.optype: 'logical', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
% 'int64', 'uint64', 'single', 'double', 'single complex' or 'double complex'.
%
% The class of z is the same as the class of the output of the operator, which
% is op.optype except for: (1) 'eq', 'ne', 'gt', 'lt', 'ge', 'le', in which
% case z is logical, (2) 'complex', where x and y are real and z is complex,
% (3) bitshift (where x has the optype and y is int8).
%
% Intrinsic built-in operators are used as much as possible, so as to test
% GraphBLAS operators.  Some must be done in GraphBLAS because the typecasting,
% divide-by-zero and overflow rules for integers differs.
%
% Positional ops are not computed by this function.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the operator name and class
[opname optype ztype xtype ytype] = GB_spec_operator (op, GB_spec_type (A)) ;

if (GB_spec_is_positional (opname))
    error ('positional op not supported by this funciton') ;
end

% cast the inputs A and B to the inputs of the operator
if (~isequal (GB_spec_type (A), xtype))
    x = GB_mex_cast (A, xtype) ;
else
    x = A ;
end

use_builtin = (isa (x, 'float') && ...
    (test_contains (optype, 'single') || test_contains (optype, 'double'))) ;

if (nargin > 2 && ~ischar (B))
    if (~isequal (GB_spec_type (B), ytype))
        y = GB_mex_cast (B, ytype) ;
    else
        y = B ;
    end
    use_builtin = use_builtin && isa (y, 'float') ;
end

switch opname

    % binary operators, result is ztype
    case 'first'
        z = x ;
    case 'second'
        z = y ;
    case 'any'
        z = y ;
    case { 'pair', 'oneb' }
        z = GB_spec_ones (size (x), ztype) ;
    case 'min'
        % min(x,y) in SuiteSparse:GraphBLAS is min(x,y,'omitnan') with built-in.
        % see discussion in SuiteSparse/GraphBLAS/Source/GB.h
        % z = min (x,y,'omitnan') ;
        z = GB_mex_op (op, x, y) ;
    case 'max'
        % z = max (x,y,'omitnan') ;
        z = GB_mex_op (op, x, y) ;
    case 'plus'
        if (use_builtin)
            z = x + y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'minus'
        if (use_builtin)
            z = x - y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'rminus'
        if (use_builtin)
            z = y - x ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'times'
        if (use_builtin)
            z = x .* y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'div'
        if (use_builtin)
            z = x ./ y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'rdiv'
        if (use_builtin)
            z = y ./ x ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'pow'
        if (use_builtin)
            z = x .^ y ;
        else
            z = GB_mex_op (op, x, y) ;
        end

    % 6 binary comparators (result is ztype)
    case 'iseq'
        z = GB_mex_cast (x == y, ztype) ;
    case 'isne'
        z = GB_mex_cast (x ~= y, ztype) ;
    case 'isgt'
        z = GB_mex_cast (x >  y, ztype) ;
    case 'islt'
        z = GB_mex_cast (x <  y, ztype) ;
    case 'isge'
        z = GB_mex_cast (x >= y, ztype) ;
    case 'isle'
        z = GB_mex_cast (x <= y, ztype) ;

    % 6 binary comparators (result is boolean)
    case 'eq'
        z = (x == y) ;
    case 'ne'
        z = (x ~= y) ;
    case 'gt'
        z = (x >  y) ;
    case 'lt'
        z = (x <  y) ;
    case 'ge'
        z = (x >= y) ;
    case 'le'
        z = (x <= y) ;

    % 6 index_unop comparators (result is boolean)
    case 'valueeq'
        z = (x == y) ;
    case 'valuene'
        z = (x ~= y) ;
    case 'valuegt'
        z = (x >  y) ;
    case 'valuelt'
        z = (x <  y) ;
    case 'valuege'
        z = (x >= y) ;
    case 'valuele'
        z = (x <= y) ;

    % 3 binary logical operators (result is ztype)
    case 'or'
        z = GB_mex_cast ((x ~= 0) | (y ~= 0), ztype) ;
    case 'and'
        z = GB_mex_cast ((x ~= 0) & (y ~= 0), ztype) ;
    case 'xor'
        z = GB_mex_cast ((x ~= 0) ~= (y ~= 0), ztype) ;

    % bitwise operators
    case { 'bitget', 'bget' }
        bits = GB_spec_nbits (ztype) ;
        m = (y > 0 & y <= bits) ;
        t = bitget (x (m), y (m), ztype) ;
        z = zeros (size (x), ztype) ;
        z (m) = t ;

    case { 'bitset', 'bset' }
        bits = GB_spec_nbits (ztype) ;
        m = (y > 0 & y <= bits) ;
        t = bitset (x (m), y (m), 1, ztype) ;
        z = x ;
        z (m) = t ;

    case { 'bitclr', 'bclr' }
        bits = GB_spec_nbits (ztype) ;
        m = (y > 0 & y <= bits) ;
        t = bitset (x (m), y (m), 0, ztype) ;
        z = x ;
        z (m) = t ;

    case { 'bitand', 'band' }
        z = bitand (x, y, ztype) ;
    case { 'bitor', 'bor' }
        z = bitor (x, y, ztype) ;
    case { 'bitxor', 'bxor' }
        z = bitxor (x, y, ztype) ;
    case { 'bitxnor', 'bxnor' }
        z = bitcmp (bitxor (x, y, ztype), ztype) ;
    case { 'bitshift', 'bshift' }
        z = bitshift (x, y, ztype) ;
    case { 'bitnot', 'bitcmp', 'bnot', 'bcmp' }
        z = bitcmp (x, ztype) ;

    case 'atan2'
        z = atan2 (x,y) ;

    case 'hypot'
        z = hypot (x,y) ;

    case { 'ldexp', 'pow2' }
        z = pow2 (x,y) ;

    case { 'fmod', 'rem' }
        % see C11 fmod function
        % the built-in rem differs slightly from the C11 fmod,
        % if x/y is O(eps) smaller than an integer.
        z = rem (x,y) ;

    case { 'remainder' }
        % see C11 remainder function
        m = (y ~= 0 & x ~= y)  ;
        z = nan (size (x), ztype) ;
        z (x == y) = 0 ;
        z (m) = x (m) - round (x (m) ./ y (m)) .* y (m) ;

    case { 'copysign' }
        % see C11 copysign function
        z = abs (x) .* (2 * double (y >= 0) - 1) ;

    case { 'complex', 'cmplx' }
        z = complex (x,y) ;

    % unary operators (result is ztype)
    case 'one'
        z = GB_mex_cast (1, ztype) ;
    case 'identity'
        z = x ;
    case 'ainv'
        if (use_builtin)
            z = -x ;
        else
            z = GB_mex_op (op, x) ;
        end
    case 'abs'
        if (use_builtin)
            z = abs (x) ;
        else
            z = GB_mex_op (op, x) ;
        end
    case 'minv'
        if (use_builtin)
            z = 1 ./ x ;
        else
            z = GB_mex_op (op, x) ;
        end
    case 'not'
        z = GB_mex_cast (~(x ~= 0), ztype) ;

    case 'bnot'
        z = bitcmp (x) ;

    case 'sqrt'
        z = sqrt (x) ;

    case 'log'
        z = log (x) ;

    case 'exp'
        z = exp (x) ;

    case 'sin'
        z = sin (x) ;

    case 'cos'
        z = cos (x) ;

    case 'tan'
        z = tan (x) ;

    case 'asin'
        z = asin (x) ;

    case 'acos'
        z = acos (x) ;

    case 'atan'
        z = atan (x) ;

    case 'sinh'
        z = sinh (x) ;

    case 'cosh'
        z = cosh (x) ;

    case 'tanh'
        z = tanh (x) ;

    case 'asinh'
        z = asinh (x) ;

    case 'acosh'
        z = acosh (x) ;

    case 'atanh'
        z = atanh (x) ;

    case { 'sign', 'signum' }
        z = sign (x) ;

    case 'ceil'
        z = ceil (x) ;

    case 'floor'
        z = floor (x) ;

    case 'round'
        z = round (x) ;

    case { 'trunc', 'fix' }
        z = fix (x) ;

    case { 'exp2' }
        z = 2.^x ;

    case 'expm1'
        z = expm1 (x) ;

    case 'log10'
        z = log10 (x) ;

    case 'log2'
        z = log2 (x) ;

    case 'log1p'
        z = log1p (x) ;

    case { 'lgamma', 'gammaln' }
        z = gammaln (x) ;

    case { 'tgamma', 'gamma' }
        z = gamma (x) ;

    case 'erf'
        z = erf (x) ;

    case 'erfc'
        z = erfc (x) ;

    case 'cbrt'
        z = nthroot (x, 3) ;

    case 'frexpx'
        [z,~] = log2 (x) ;

    case 'frexpe'
        [~,z] = log2 (x) ;

    case 'conj'
        z = conj (x) ;

    case { 'creal', 'real' }
        z = real (x) ;

    case { 'cimag', 'imag' }
        z = imag (x) ;

    case { 'carg', 'angle' }
        z = angle (x) ;

    case 'isinf'
        z = isinf (x) ;

    case 'isnan'
        z = isnan (x) ;

    case 'isfinite'
        z = isfinite (x) ;

    otherwise
        opname
        error ('unknown op') ;
end

if (~isequal (ztype, GB_spec_type (z)))
    z = GB_mex_cast (z, ztype) ;
end

C = z ;

