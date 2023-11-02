function codegen_red_method (opname, op, atype, identity, terminal, panel)
%CODEGEN_RED_METHOD create a reduction function, C = reduce (A)
%
% codegen_red_method (opname, op, atype, identity, terminal)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

[aname, unsigned, bits] = codegen_type (atype) ;

name = sprintf ('%s_%s', opname, aname) ;
is_any = isequal (opname, 'any') ;

% function names
fprintf (f, 'm4_define(`_bld'', `_bld__%s'')\n', name) ;

% the type of A, S, T, X, Y, and Z (no typecasting)
ztype = atype ;
fprintf (f, 'm4_define(`GB_atype'', `#define GB_A_TYPE %s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_stype'', `#define GB_S_TYPE %s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ttype'', `#define GB_T_TYPE %s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_xtype'', `#define GB_X_TYPE %s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ytype'', `#define GB_Y_TYPE %s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ztype'', `#define GB_Z_TYPE %s'')\n', atype) ;

fprintf (f, 'm4_define(`GB_atype_parameter'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_stype_parameter'', `%s'')\n', atype) ;
fprintf (f, 'm4_define(`GB_ttype_parameter'', `%s'')\n', atype) ;

is_monoid = ~isempty (identity) ;
if (is_monoid)
    % monoid function name and identity value
    fprintf (f, 'm4_define(`_red'',    `_red__%s'')\n', name) ;
    fprintf (f, 'm4_define(`GB_identity'', `%s'')\n', identity) ;
else
    % first and second operators are not monoids
    fprintf (f, 'm4_define(`_red'',    `_red__(none)'')\n') ;
    fprintf (f, 'm4_define(`GB_identity'', `(none)'')\n') ;
end

% identity value for the monoid
if (is_monoid)
    define_id = sprintf (' %s z = %s', ztype, identity) ;
    define_const_id = sprintf (' const %s z = %s', ztype, identity) ;
    fprintf (f, 'm4_define(`GB_declare_identity'', `#define GB_DECLARE_IDENTITY(z)%s'')\n', define_id) ;
    fprintf (f, 'm4_define(`GB_declare_const_identity'', `#define GB_DECLARE_IDENTITY_CONST(z)%s'')\n', define_const_id) ;
end

% A is never iso, so GBX is not needed
fprintf (f, 'm4_define(`GB_declarea'', `#define GB_DECLAREA(aij) %s aij'')\n', atype) ;
fprintf (f, 'm4_define(`GB_geta'', `#define GB_GETA(aij,Ax,pA,A_iso) aij = Ax [pA]'')\n') ;

tvalue = '' ;
tbreak = '' ;
tcondition = '' ;

if (is_any)
    % ANY monoid is terminal
    terminal = ' ' ;
    is_terminal = 1 ;
elseif (~isempty (terminal))
    % monoid is terminal
    is_terminal = 1 ;
    tbreak = sprintf (' if (z == %s) { break ; }', terminal) ;
    tvalue = sprintf (' const %s zterminal = %s', ztype, terminal) ;
    tcondition = sprintf (' (z == %s)', terminal) ;
else
    % monoid is not terminal
    is_terminal = 0 ;
end

if (is_any)
    fprintf (f, 'm4_define(`GB_is_any_monoid'', `%s'')\n', ...
        '#define GB_IS_ANY_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_any_monoid'', `'')\n') ;
end

if (is_terminal)
    fprintf (f, 'm4_define(`GB_monoid_is_terminal'', `%s'')\n', ...
        '#define GB_MONOID_IS_TERMINAL 1') ;
else
    fprintf (f, 'm4_define(`GB_monoid_is_terminal'', `'')\n') ;
end

if (~isempty (tcondition))
    % monoid is terminal
    fprintf (f, 'm4_define(`GB_terminal_condition'', `#define GB_TERMINAL_CONDITION(z,zterminal)%s'')\n', tcondition) ;
    fprintf (f, 'm4_define(`GB_if_terminal_break'', `#define GB_IF_TERMINAL_BREAK(z,zterminal)%s'')\n', tbreak) ;
    fprintf (f, 'm4_define(`GB_declare_const_terminal'', `#define GB_DECLARE_TERMINAL_CONST(zterminal)%s'')\n', tvalue) ;
else
    % will be defined by GB_monoid_shared_definitions.h
    fprintf (f, 'm4_define(`GB_terminal_condition'', `'')\n') ;
    fprintf (f, 'm4_define(`GB_if_terminal_break'', `'')\n') ;
    fprintf (f, 'm4_define(`GB_declare_const_terminal'', `'')\n') ;
end

if (is_any)
    % no panel for the ANY monoid
    fprintf (f, 'm4_define(`GB_panel'', `'')\n') ;
else
    fprintf (f, 'm4_define(`GB_panel'', `#define GB_PANEL %d'')\n', panel) ;
end

% create the update operator
update_op = op {1} ;
update_op = strrep (update_op, 'zarg', 'z') ;
update_op = strrep (update_op, 'yarg', 'a') ;
fprintf (f, 'm4_define(`GB_update_op'', `#define GB_UPDATE(z,a)  %s'')\n', update_op) ;

% create the geta_and_update macro
geta_and_update = op {1} ;
geta_and_update = strrep (geta_and_update, 'zarg', 'z') ;
geta_and_update = strrep (geta_and_update, 'yarg', 'Ax [p]') ;
fprintf (f, 'm4_define(`GB_geta_and_update'', `#define GB_GETA_AND_UPDATE(z,Ax,p) %s'')\n', geta_and_update) ;

% create the dup operator
dup_op = op {1} ;
dup_op = strrep (dup_op, 'zarg', 'Tx [k]') ;
dup_op = strrep (dup_op, 'yarg', 'Sx [i]') ;
fprintf (f, 'm4_define(`GB_bld_dup'', `#define GB_BLD_DUP(Tx,k,Sx,i)  %s'')\n', dup_op) ;

% create the function operator
add_op = op {2} ;
add_op = strrep (add_op, 'zarg', 'z') ;
add_op = strrep (add_op, 'xarg', 'zin') ;
add_op = strrep (add_op, 'yarg', 'a') ;
fprintf (f, 'm4_define(`GB_add_op'', `#define GB_ADD(z,zin,a) %s'')\n', add_op) ;

% create the disable flag
disable  = sprintf ('defined(GxB_NO_%s)', upper (opname)) ;
disable = [disable (sprintf (' || defined(GxB_NO_%s)', upper (aname)))] ;
disable = [disable (sprintf (' || defined(GxB_NO_%s_%s)', upper (opname), upper (aname)))] ;
fprintf (f, 'm4_define(`GB_disable'', `#if (%s)\n#define GB_DISABLE 1\n#else\n#define GB_DISABLE 0\n#endif\n'')\n', disable) ;

fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

if (is_monoid)
    % construct the *.c file for the reduction to scalar
    cmd = sprintf ('cat control.m4 Generator/GB_red.c | m4 -P | awk -f codegen_blank.awk > FactoryKernels/GB_red__%s.c', name) ;
    fprintf ('.') ;
    system (cmd) ;
    % append to the *.h file
    system ('cat control.m4 Generator/GB_red.h | m4 -P | awk -f codegen_blank.awk | grep -v SPDX >> FactoryKernels/GB_red__include.h') ;
end

% construct the build *.c and *.h files
cmd = sprintf ('cat control.m4 Generator/GB_bld.c | m4 -P | awk -f codegen_blank.awk > FactoryKernels/GB_bld__%s.c', name) ;
fprintf ('.') ;
system (cmd) ;
system ('cat control.m4 Generator/GB_bld.h | m4 -P | awk -f codegen_blank.awk | grep -v SPDX >> FactoryKernels/GB_bld__include.h') ;

delete ('control.m4') ;

