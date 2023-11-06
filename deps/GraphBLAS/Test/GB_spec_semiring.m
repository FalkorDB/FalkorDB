function [multiply_op add_op identity ztype xtype ytype] = GB_spec_semiring (semiring)
%GB_SPEC_SEMIRING create a semiring
%
% [multiply_op add_op identity ztype xtype ytype] = GB_spec_semiring (semiring)
%
% Given a semiring, extract the multiply operator, additive operator, additive
% identity, and the ztype of z for z=multiply(...) and the monoid z=add(z,z).
%
% A semiring is a struct with 3 fields, each a string, with defaults used if
% fields are not present.  None of the content of the semiring should be
% a struct.
%
%   multiply    a string with the name of the 'multiply' binary operator
%               (default is 'times').
%
%   add         a string with the name of the 'add' operator (default is 'plus')
%               or a monoid struct with add.opname and add.optype.
%
%   class       the built-in class of the operators (default is 'double',
%               unless the multiply operator is 'or', 'and, or 'xor').  Any
%               logical or numeric class is supported, which are the same as
%               the 11 built-in GraphBLAS types: 'logical' (boolean in
%               GraphBLAS), 'int8', 'uint8', 'int16', 'uint16', 'int32',
%               'uint32', 'int64', 'uint64', 'single' (FP43 in GraphBLAS),
%               'double' (FP64 in GraphBLAS), 'single complex', and 'double
%               complex'

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% set the default semiring
if (isempty (semiring))
    semiring = struct ;
end
if (~isfield (semiring, 'multiply'))
    semiring.multiply = 'times' ;
end
if (~isfield (semiring, 'add'))
    semiring.add = 'plus' ;
end
if (~isfield (semiring, 'class'))
    semiring.class = 'double' ;
end

% create the multiply operator.  No error checks; it will be checked later
% and can be any valid GraphBLAS binary operator.
[mult mult_optype ztype xtype ytype] = GB_spec_operator (semiring.multiply, semiring.class);
multiply_op.opname =  mult ;
multiply_op.optype = mult_optype ;

if (isequal (ytype, 'none'))
    error ('invalid multiply op') ;
end

% create the add operator
[add_opname add_optype add_ztype add_xtype add_ytype] = GB_spec_operator (semiring.add, ztype) ;
add_op.opname = add_opname ;
add_op.optype = add_optype ;

% get the identity of the add operator
identity = GB_spec_identity (add_op) ;

% make sure the monoid is valid
if (~isequal (add_ztype, add_xtype) || ~isequal (add_ztype, add_xtype))
    error ('invalid monoid') ;
end

switch (add_opname)
    case { 'min', 'max', 'plus', 'times', 'any', ...
        'or', 'and', 'xor', 'eq', ...
        'lor', 'land', 'lxor', 'lnxor', ...
        'bitor', 'bitand', 'bitxor', 'bitxnor', ...
        'bor', 'band', 'bxor', 'bxnor' }
        % valid monoid
    otherwise
        error ('invalid monoid') ;
end

% make sure the monoid matches the operator ztype
if (~isequal (add_ztype, ztype))
    error ('invalid monoid: must match ztype of multiplier') ;
end

%   semiring
%   multiply_op
%   add_op
%   identity
%   ztype
%   xtype
%   ytype

% make sure the semiring is built-in
try
    GB_mex_semiring (semiring, 0) ;
catch
    error ('not builtin semiring: %s.%s.%s', add_opname, mult, xtype) ;
end

