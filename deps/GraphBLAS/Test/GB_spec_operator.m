function [opname optype ztype xtype ytype] = GB_spec_operator (op,optype_default)
%GB_SPEC_OPERATOR get the contents of an operator
%
% On input, op can be a struct with a string op.opname that gives the operator
% name, and a string op.optype with the operator type.  Alternatively, op can
% be a string with the operator name, in which case the operator type is given
% by optype_default.
%
% ztype, xtype, and ytype are the types of z, x, and y for z = f(x,y), if
% f is a binary operator, or z = f(x) if f is a unary operator.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isempty (op))
    % No operator has been defined; return an empty operator.  GB_spec_accum
    % uses this condition just like the (accum == NULL) condition in the C
    % version of GraphBLAS.  It means C<Mask>=T is to be done instead of
    % C<Mask>=accum(C,T).
    opname = '' ;
    optype = '' ;
    xtype = '' ;
    ytype = '' ;
    ztype = '' ;
    return
elseif (isstruct (op))
    % op is a struct with opname and optype
    opname = op.opname ;
    optype = op.optype ;
else
    % op is a string
    opname = op ;
    if (nargin == 1 && GB_spec_is_positional (opname))
        % optype_default is ignored
        optype = 'int64' ;
    else
        optype = optype_default ;
    end
end

% xtype is always the optype
xtype = optype ;

% for binary ops: ytype is usually the optype, except for bitshift.
% for unary ops:  ytype is 'none'.
ytype = optype ;

% ztype is usually the optype, except for the cases below
ztype = optype ;

is_float   = test_contains (optype, 'single') || test_contains (optype, 'double') ;
is_complex = test_contains (optype, 'complex') ;
is_int     = test_contains (optype, 'int') ; % int or uint
is_logical = isequal (optype, 'logical') ;
is_real_float = is_float && ~is_complex ;

%-------------------------------------------------------------------------------
% boolean rename:
%-------------------------------------------------------------------------------

if (is_logical)

    switch (opname)

        case { 'div' }
            opname = 'first' ;

        case { 'rdiv' }
            opname = 'second' ;

        case { 'min', 'times' }
            opname = 'and' ;

        case { 'max', 'plus' }
            opname = 'or' ;

        case { 'minus', 'rminus', 'isne', 'ne' }
            opname = 'xor' ;

        case { 'iseq' }
            opname = 'eq' ;

        case { 'isgt' }
            opname = 'gt' ;

        case { 'islt' }
            opname = 'lt' ;

        case { 'isge', 'pow' }
            opname = 'ge' ;

        case { 'isle' }
            opname = 'le' ;

        otherwise
            % op not renamed
    end
end

%-------------------------------------------------------------------------------
% get the x,y,z types of the operator and check if valid
%-------------------------------------------------------------------------------

switch opname

    %--------------------------------------------------------------------------
    % binary ops for all 13 types
    %--------------------------------------------------------------------------

    case { 'first', 'second', 'pow', 'plus', 'minus', 'times', 'div', ...
           'rminus', 'rdiv', 'pair', 'oneb', 'any', 'iseq', 'isne' }
        % x,y,z types are all the same.

    case { 'eq', 'ne' }
        % x,y types the the same, z is logical
        ztype = 'logical' ;

    %--------------------------------------------------------------------------
    % binary ops for 11 types (all but complex)
    %--------------------------------------------------------------------------

    case { 'max', 'min', 'isgt', 'islt', 'isge', 'isle', 'or', 'and', 'xor' }
        % x,y,z types are all the same.
        % available for 11 real types, not complex
        if (is_complex)
            error ('invalid op') ;
        end

    case { 'gt', 'lt', 'ge', 'le' }
        % x,y types the the same, z is logical
        % available for 11 real types, not complex
        ztype = 'logical' ;
        if (is_complex)
            error ('invalid op') ;
        end

    %--------------------------------------------------------------------------
    % binary ops for real floating point
    %--------------------------------------------------------------------------

    case { 'atan2', 'hypot', 'fmod', 'remainder', 'ldexp', 'copysign' }
        % x,y,z types are all the same.
        % available for real float and double only
        if (~is_real_float)
            error ('invalid op') ;
        end

    case { 'cmplx', 'complex' }
        % x and y are real (float or double).  z is the corresponding complex
        if (~is_real_float)
            error ('invalid op') ;
        elseif (isequal (optype, 'single'))
            ztype = 'single complex' ;
        else
            ztype = 'double complex' ;
        end
        opname = 'cmplx' ;

    %--------------------------------------------------------------------------
    % binary ops for integer only
    %--------------------------------------------------------------------------

    case { 'bitor', 'bitand', 'bitxor', 'bitxnor', ...
           'bitget', 'bitset', 'bitclr', ...
           'bor', 'band', 'bxor', 'bxnor', ...
           'bget', 'bset', 'bclr' }

        % x,y,z types are all the same.
        % available for int and uint only
        if (~(is_int))
            error ('invalid op') ;
        end

        switch (opname)
            case { 'bitor', 'bor' }
                opname = 'bor' ;
            case { 'bitand', 'band' }
                opname = 'band' ;
            case { 'bitxor', 'bxor' }
                opname = 'bxor' ;
            case { 'bitxnor', 'bxnor' }
                opname = 'bxnor' ;
            case { 'bitget', 'bget' }
                opname = 'bget' ;
            case { 'bitset', 'bset' }
                opname = 'bset' ;
            case { 'bitclr', 'bclr' }
                opname = 'bclr' ;
        end

    case { 'bitshift' , 'bshift' }
        % x,z types are the same.  y is int8
        % available for int and uint only
        ytype = 'int8' ;
        if (~(is_int))
            error ('invalid op') ;
        end

    %--------------------------------------------------------------------------
    % unary ops for all 13 types
    %--------------------------------------------------------------------------

    case { 'identity', 'one', 'ainv', 'minv' }
        % x,y,z types are all the same.

    case { 'abs' }
        % x,y the same.  z is the same as x, except if x is complex
        if (isequal (optype, 'single complex'))
            ztype = 'single' ;
        elseif (isequal (optype, 'double complex'))
            ztype = 'double' ;
        else
            ztype = xtype ;
        end

    %--------------------------------------------------------------------------
    % unary ops for 11 real types
    %--------------------------------------------------------------------------

    case { 'not' }
        % x and z have the same type
        if (is_complex)
            error ('invalid op') ;
        end
        ytype = 'none' ;

    %--------------------------------------------------------------------------
    % unary ops for integer only
    %--------------------------------------------------------------------------

    case { 'bitnot', 'bnot', 'bitcmp', 'bcmp' }
        % x and z have the same type
        if (~is_int)
            error ('invalid op') ;
        end
        ytype = 'none' ;
        opname = 'bnot' ;

    %--------------------------------------------------------------------------
    % unary ops for floating-point only (both real and complex)
    %--------------------------------------------------------------------------

    case { 'sqrt',     'log',      'exp',           'log2',    ...
           'sin',      'cos',      'tan',                      ...
           'acos',     'asin',     'atan',                     ...
           'sinh',     'cosh',     'tanh',                     ...
           'acosh',    'asinh',    'atanh',                    ...
           'ceil',     'floor',    'round',         'trunc',   ...
           'exp2',     'expm1',    'log10',         'log1p',   ...
           'signum' }
        % x and z have the same type
        if (~is_float)
            error ('invalid op') ;
        end
        ytype = 'none' ;

    case { 'isinf', 'isnan', 'isfinite' }
        % z is logical
        ztype = 'logical' ;
        if (~is_float)
            error ('invalid op') ;
        end
        ytype = 'none' ;

    %--------------------------------------------------------------------------
    % unary ops for real floating-point only
    %--------------------------------------------------------------------------

    case { 'lgamma', 'tgamma', 'erf', 'erfc', 'frexpx',  'frexpe', 'cbrt' }
        % x and z have the same type
        if (~is_real_float)
            error ('invalid op') ;
        end
        ytype = 'none' ;

    %--------------------------------------------------------------------------
    % unary ops for complex only
    %--------------------------------------------------------------------------

    case { 'conj' }
        if (~is_complex)
            error ('invalid op') ;
        end
        ytype = 'none' ;

    case { 'real', 'imag', 'carg' }
        if (~is_complex)
            error ('invalid op') ;
        end
        if (isequal (optype, 'single complex'))
            ztype = 'single' ;
        else
            ztype = 'double' ;
        end
        ytype = 'none' ;

    %--------------------------------------------------------------------------
    % binary positional ops
    %--------------------------------------------------------------------------

    case { 'firsti' , 'firsti1' , 'firstj' , 'firstj1', ...
           'secondi', 'secondi1', 'secondj', 'secondj1' } ;
        if (~(isequal (ztype, 'int64') || isequal (ztype, 'int32')))
            error ('invalid op') ;
        end
        xtype = optype ;
        ytype = optype ;

    %--------------------------------------------------------------------------
    % unary positional ops
    %--------------------------------------------------------------------------

    case { 'positioni', 'positioni1', 'positionj', 'positionj1' }
        if (~(isequal (ztype, 'int64') || isequal (ztype, 'int32')))
            error ('invalid op') ;
        end
        ytype = optype ;

    %--------------------------------------------------------------------------
    % idxunop
    %--------------------------------------------------------------------------

    case { 'rowindex', 'colindex', 'diagindex' }
        s = strcmp (optype, 'int64') || strcmp (optype, 'int32') ;
        if (~s)
            error ('invalid op') ;
        end
        xtype = '' ;
        ytype = optype ;
        ztype = optype ;

    case { 'tril', 'triu', 'diag', 'offdiag', ...
        'colle', 'colgt', 'rowle', 'rowgt' }
        s = strcmp (optype, 'int64') ;
        if (~s)
            error ('invalid op') ;
        end
        xtype = '' ;
        ytype = optype ;
        ztype = 'logical' ;

    case { 'valuene', 'valueeq' }
        s = true ;
        xtype = optype ;
        ytype = optype ;
        ztype = 'logical' ;

    case { 'valuelt', 'valuele', 'valuegt', 'valuege' }
        s = ~test_contains (optype, 'complex') ;
        if (~s)
            error ('invalid op') ;
        end
        xtype = optype ;
        ytype = optype ;
        ztype = 'logical' ;

    otherwise
        error ('unknown op') ;

end

