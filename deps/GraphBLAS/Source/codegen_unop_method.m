function codegen_unop_method (unop, op, ztype, xtype)
%CODEGEN_UNOP_METHOD create a function to compute C=unop(A)
%
% codegen_unop_method (unop, op, ztype, xtype)
%
%   unop: the name of the operator
%   op: a string defining the computation
%   ztype: the type of z for z=f(x)
%   xtype: the type of x for z=f(x)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

[zname, zunsigned, zbits] = codegen_type (ztype) ;
[xname, xunsigned, xbits] = codegen_type (xtype) ;

name = sprintf ('%s_%s_%s', unop, zname, xname) ;

% determine if the op is identity with no typecast
is_identity = isequal (unop, 'identity') ;
no_typecast = isequal (ztype, xtype) ;
if (is_identity && no_typecast)
    % disable the _unop_apply method
    fprintf (f, 'm4_define(`_unop_apply'', `_unop_apply__(none)'')\n') ;
    fprintf (f, 'm4_define(`if_unop_apply_enabled'', `-1'')\n') ;
else
    fprintf (f, 'm4_define(`_unop_apply'', `_unop_apply__%s'')\n', name) ;
    fprintf (f, 'm4_define(`if_unop_apply_enabled'', `0'')\n') ;
end

% function names
fprintf (f, 'm4_define(`_unop_tran'', `_unop_tran__%s'')\n', name) ;

% type of C and A
fprintf (f, 'm4_define(`GB_ctype'', `#define GB_C_TYPE %s'')\n', ztype) ;
fprintf (f, 'm4_define(`GB_atype'', `#define GB_A_TYPE %s'')\n', xtype) ;
fprintf (f, 'm4_define(`GB_xtype'', `#define GB_X_TYPE %s'')\n', xtype) ;
fprintf (f, 'm4_define(`GB_ztype'', `#define GB_Z_TYPE %s'')\n', ztype) ;

A_is_pattern = (isempty (strfind (op, 'xarg'))) ;

% to get an entry from A
if (A_is_pattern)
    % A(i,j) is not needed
    fprintf (f, 'm4_define(`GB_declarea'', `#define GB_DECLAREA(aij)'')\n') ;
    fprintf (f, 'm4_define(`GB_geta'', `#define GB_GETA(aij,Ax,pA,A_iso)'')\n') ;
else
    % A is never iso
    fprintf (f, 'm4_define(`GB_declarea'', `#define GB_DECLAREA(aij) %s aij'')\n', xtype) ;
    fprintf (f, 'm4_define(`GB_geta'', `#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]'')\n') ;
end

% type-specific iminv
if (~isempty (strfind (op, 'iminv')))
    if (zunsigned)
        op = strrep (op, 'iminv (', sprintf ('idiv_uint%d (1, ', zbits)) ;
    else
        op = strrep (op, 'iminv (', sprintf ('idiv_int%d (1, ', zbits)) ;
    end
end

% create the unary operator
op = strrep (op, 'xarg', 'x') ;
fprintf (f, 'm4_define(`GB_unaryop'', `#define GB_UNARYOP(z,x) z = %s'')\n', op) ;

% create the disable flag
disable  = sprintf ('defined(GxB_NO_%s)', upper (unop)) ;
disable = [disable (sprintf (' || defined(GxB_NO_%s)', upper (zname)))] ;
if (~isequal (zname, xname))
    disable = [disable (sprintf (' || defined(GxB_NO_%s)', upper (xname)))] ;
end
fprintf (f, 'm4_define(`GB_disable'', `#if (%s)\n#define GB_DISABLE 1\n#else\n#define GB_DISABLE 0\n#endif\n'')\n', disable) ;
fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

% construct the *.c file
cmd = sprintf ('cat control.m4 Generator/GB_unop.c | m4 -P | awk -f codegen_blank.awk > FactoryKernels/GB_unop__%s.c', name) ;
fprintf ('.') ;
system (cmd) ;

% append to the *.h file
system ('cat control.m4 Generator/GB_unop.h | m4 -P | awk -f codegen_blank.awk | grep -v SPDX >> FactoryKernels/GB_unop__include.h') ;

delete ('control.m4') ;

