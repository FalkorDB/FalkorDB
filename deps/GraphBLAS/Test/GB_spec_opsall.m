function [binops unary_ops add_ops types semirings selops idxunop] = GB_spec_opsall
%GB_SPEC_OPSALL return a list of all operators, types, and semirings
%
% [binops unary_ops add_ops types semirings select_ops idxunop] = GB_spec_opsall

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% types
%-------------------------------------------------------------------------------

% all 13 built-in types
types.all = {
'logical', ...
'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', ...
'single', 'double', 'single complex', 'double complex' } ;

% all but complex
types.real = {
'logical', ...
'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', ...
'single', 'double' } ;

% integer types
types.int = {
'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64' } ;

% floating-point types
types.float = { 'single', 'double', 'single complex', 'double complex'} ;

% floating-point real
types.fpreal = { 'single', 'double' } ;

% complex
types.complex = { 'single complex', 'double complex'} ;

%-------------------------------------------------------------------------------
% binary ops
%-------------------------------------------------------------------------------

% binary operators for all 13 types
binops.alltypes = {
'first',     % z = x
'second',    % z = y
'pair',      % z = 1
'plus',      % z = x + y
'minus',     % z = x - y
'rminus',    % z = y - x
'times',     % z = x * y
'div',       % z = x / y
'rdiv',      % z = y / x
'iseq',      % z = (x == y)
'isne',      % z = (x != y)
'eq',        % z = (x == y)
'ne',        % z = (x != y)
'pow',       % z = x.^y
'any',       % z = any(x,y)
'oneb'}' ;   % z = 1 (same as pair)

% binary operators for 11 types (all but complex)
binops.real = {
'min',       % z = min(x,y)
'max',       % z = max(x,y)
'isgt',      % z = (x >  y)
'islt',      % z = (x <  y)
'isge',      % z = (x >= y)
'isle',      % z = (x <= y)
'gt',        % z = (x >  y)
'lt',        % z = (x <  y)
'ge',        % z = (x >= y)
'le',        % z = (x <= y)
'or',        % z = x || y
'and',       % z = x && y
'xor'        % z = x != y
}' ;

% binary operators for integer types only
binops.int = {
    'bor', 'band', 'bxor', 'bxnor', ...
    'bget', 'bset', 'bclr', 'bshift' } ;

% binary operators for floating-point only (FP32, FP64, FC32, FC64)
binops.float = { } ;

% binary ops for FP32 and FP64 only
binops.fpreal = {
    'atan2', 'hypot', 'fmod', 'remainder', 'ldexp', 'copysign', 'cmplx' } ;

% binary ops for FC32 and FC64 only
binops.complex = { } ;

% binary positional ops
binops.positional = { 'firsti' , 'firsti1' , 'firstj' , 'firstj1', ...
                      'secondi', 'secondi1', 'secondj', 'secondj1' } ;

% list of all binary ops
binops.all = [ binops.alltypes, binops.real, binops.int, ...
    binops.float, binops.fpreal, binops.complex, binops.positional ] ;

%-------------------------------------------------------------------------------
% unary ops
%-------------------------------------------------------------------------------

% defined for all 13 types
unary_ops.alltypes = {
'one',       % z = 1
'identity',  % z = x
'ainv',      % z = -x
'abs',       % z = abs(x)       (z is always real)
'minv',      % z = 1/x
}' ;

% unary ops for 11 real types only (all but FC32 and FC64)
unary_ops.real = {
'not'        % z = ~x           
} ;

% unary ops for 8 integer types only (INT* and UINT*)
unary_ops.int = {
'bnot'     % z = ~x           
} ;

% unary ops for floating-point only (FP32, FP64, FC32, FC64)
unary_ops.float = {
    'sqrt',     'log',      'exp',           'log2',    ...
    'sin',      'cos',      'tan',                      ...
    'acos',     'asin',     'atan',                     ...
    'sinh',     'cosh',     'tanh',                     ...
    'acosh',    'asinh',    'atanh',                    ...
    'ceil',     'floor',    'round',         'trunc',   ...
    'exp2',     'expm1',    'log10',         'log1p',   ...
    'isinf',    'isnan',    'isfinite',      'signum'   } ;

% unary ops for FP32 and FP64 only
unary_ops.fpreal = {
'lgamma', 'tgamma', 'erf', 'erfc', 'frexpx',  'frexpe', 'cbrt' } ;

% unary ops for FC32 and FC64 only
unary_ops.complex = {
    'conj', 'real', 'imag', 'carg' } ;

% unary positional ops
unary_ops.positional = { 'positioni', 'positioni1', 'positionj', 'positionj1' };

% list of all unary ops
unary_ops.all = [ unary_ops.alltypes, unary_ops.real, unary_ops.int, ...
    unary_ops.float, unary_ops.fpreal, unary_ops.complex, ...
    unary_ops.positional ] ;

%-------------------------------------------------------------------------------
% valid binary ops
%-------------------------------------------------------------------------------

add_ops = { ...
    'min', 'max', ...                           % 11 real types only
    'plus', 'times', 'any', ...                 % all 13 types
    'or', 'and', 'xor', 'eq', ...               % just boolean
    'bor', 'band', 'bxor', 'bxnor' } ;          % just integer

nonbool = {
'int8' 
'int16' 
'int32'
'int64'
'uint8' 
'uint16'
'uint32'
'uint64'
'single'
'double'
} ;

%-------------------------------------------------------------------------------
% create all unique semirings using built-in operators
%-------------------------------------------------------------------------------

n = 0 ;

%-------------------------------------------------------------------------------
% 1000: x,y,z all nonboolean:  20*5*10
%-------------------------------------------------------------------------------

for mult = {'first', 'second', 'oneb', 'min', 'max', 'plus', 'minus', ...
            'rminus', 'times', 'div', 'rdiv', ...
            'iseq', 'isne', 'isgt', 'islt', 'isge', 'isle', ...
            'or', 'and', 'xor', }
    for add = { 'min', 'max', 'plus', 'times', 'any' }
        for c = nonbool'
            n = n + 1 ;
            s = struct ('multiply', mult{1}, 'add', add{1}, 'class', c{1}) ;
            semirings {n} = s ;
            % fprintf ('%3d %s-%s-%s\n', n, add{1}, mult{1}, c{1}) ;
        end
    end
end

%-------------------------------------------------------------------------------
% 300: x,y nonboolean, z boolean: 6 * 5 * 10
%-------------------------------------------------------------------------------

for mult = { 'eq', 'ne', 'gt', 'lt', 'ge', 'le' }
    for add = { 'or', 'and', 'xor', 'eq', 'any' }
        for c = nonbool'
            n = n + 1 ;
            s = struct ('multiply', mult{1}, 'add', add{1}, 'class', c{1}) ;
            semirings {n} = s ;
            % fprintf ('%3d %s-%s-%s\n', n, add{1}, mult{1}, c{1}) ;
        end
    end
end

%-------------------------------------------------------------------------------
% 55: x,y,z all boolean: 11 * 5
%-------------------------------------------------------------------------------

for mult = { 'first', 'second', 'oneb', 'or', 'and', 'xor', ...
    'eq', 'gt', 'lt', 'ge', 'le' }
    for add = { 'or', 'and', 'xor', 'eq', 'any' }
        n = n + 1 ;
        s = struct ('multiply', mult{1}, 'add', add{1}, 'class', c{1}) ;
        semirings {n} = s ;
        % fprintf ('%3d %s-%s-logical\n', n, add{1}, mult{1}) ;
    end
end

%-------------------------------------------------------------------------------
% 64: bitwise
%-------------------------------------------------------------------------------

for mult = { 'bor', 'band', 'bxor', 'bxnor' }
    for add = { 'bor', 'band', 'bxor', 'bxnor' }
        for c = { 'uint8', 'uint16', 'uint32', 'uint64' } ;
            n = n + 1 ;
            s = struct ('multiply', mult{1}, 'add', add{1}, 'class', c{1}) ;
            semirings {n} = s ;
            % fprintf ('%3d %s-%s-logical\n', n, add{1}, mult{1}) ;
        end
    end
end

%-------------------------------------------------------------------------------
% 54: complex
%-------------------------------------------------------------------------------

for mult = {'first', 'second', 'oneb', 'plus', 'minus', ...
            'rminus', 'times', 'div', 'rdiv' }
    for add = { 'plus', 'times', 'any' }
        for c = { 'single complex', 'double complex' }
            n = n + 1 ;
            s = struct ('multiply', mult{1}, 'add', add{1}, 'class', c{1}) ;
            semirings {n} = s ;
            % fprintf ('%3d %s-%s-%s\n', n, add{1}, mult{1}, c{1}) ;
        end
    end
end

%-------------------------------------------------------------------------------
% 40: positional
%-------------------------------------------------------------------------------

for mult = { 'firsti' , 'firsti1' , 'firstj' , 'firstj1', ...
              'secondi', 'secondi1', 'secondj', 'secondj1' } ;
    for add = { 'min', 'max', 'plus', 'times', 'any' }
        n = n + 1 ;
        c = { 'int64' } ;
        s = struct ('multiply', mult{1}, 'add', add{1}, 'class', c{1}) ;
        semirings {n} = s ;
    end
end

%-------------------------------------------------------------------------------
% select operators
%-------------------------------------------------------------------------------

selops = { 'tril', 'triu', 'diag', 'offdiag', ...
    'nonzero',  'eq_zero',  'gt_zero',  'ge_zero',  'lt_zero',  'le_zero', ...
    'ne_thunk', 'eq_thunk', 'gt_thunk', 'ge_thunk', 'lt_thunk', 'le_thunk' }' ;

% fprintf ('semirings: %d\n', n) ;

%-------------------------------------------------------------------------------
% idxunop
%-------------------------------------------------------------------------------

idxunop = { 'rowindex', 'colindex', 'diagindex', ...
    'tril', 'triu', 'diag', 'offdiag', ...
    'colle', 'colgt', 'rowle', 'rowgt', ...
    'valuene', 'valueeq', 'valuelt', 'valuele', 'valuegt', 'valuege' } ;

