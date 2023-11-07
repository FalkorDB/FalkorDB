function codegen_unop_template (unop, bfunc, ifunc, ufunc, ffunc, dfunc, ...
    fcfunc, dcfunc)
%CODEGEN_UNOP_TEMPLATE create unop functions
%
% codegen_unop_template (unop, bfunc, ifunc, ufunc, ffunc, dfunc, ...
%       fcfunc, dcfunc)
%
%       unop:   operator name
%
%   strings defining each function, or empty if no such unary operator:
%
%       bfunc:  bool
%       ifunc:  int8, int16, int32, int64
%       ufunc:  uint8, uint16, uint32, uint64
%       ffunc:  float
%       dfunc:  double
%       fcfunc: GxB_FC32_t
%       dcfunc: GxB_FC64_t
%
% Generate functions for a unary operator, for all types.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n%-9s', unop) ;

types = { 
'bool',
'int8_t',
'int16_t',
'int32_t',
'int64_t',
'uint8_t',
'uint16_t',
'uint32_t',
'uint64_t',
'float',
'double',
'GxB_FC32_t',
'GxB_FC64_t' } ;

ntypes = length (types) ;

for code1 = 1:ntypes
    ctype = types {code1} ;

    % determine the function
    if (isequal (ctype, 'bool')) 
        func = bfunc ;
    elseif (ctype (1) == 'i')
        func = ifunc ;
    elseif (ctype (1) == 'u')
        func = ufunc ;
    elseif (isequal (ctype, 'float')) 
        func = ffunc ;
    elseif (isequal (ctype, 'double')) 
        func = dfunc ;
    elseif (isequal (ctype, 'GxB_FC32_t')) 
        func = fcfunc ;
    elseif (isequal (ctype, 'GxB_FC64_t')) 
        func = dcfunc ;
    end

    if (isempty (func))
        % skip this operator
        continue ;
    end

    codegen_unop_method (unop, func, ctype, ctype) ;
end

