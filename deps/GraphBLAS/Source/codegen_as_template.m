function codegen_as_template (xtype)
%CODEGEN_AS_TEMPLATE create a function for subassign/assign with no accum
%
% codegen_as_template (xtype)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

[fname, unsigned, bits] = codegen_type (xtype) ;
fprintf ('%-11s:  fname: %-7s  unsigned: %d bits: %d\n', xtype, fname, unsigned, bits) ;

% function names
fprintf (f, 'm4_define(`_subassign_05d'', `_subassign_05d__%s'')\n', fname) ;
fprintf (f, 'm4_define(`_subassign_06d'', `_subassign_06d__%s'')\n', fname) ;
fprintf (f, 'm4_define(`_subassign_25'', `_subassign_25__%s'')\n', fname) ;

fprintf (f, 'm4_define(`GB_ctype'', `#define GB_C_TYPE %s'')\n', xtype) ;
fprintf (f, 'm4_define(`GB_atype'', `#define GB_A_TYPE %s'')\n', xtype) ;
fprintf (f, 'm4_define(`GB_declarec'', `#define GB_DECLAREC(cwork) %s cwork'')\n', xtype) ;

% to copy a scalar into C (no typecasting)
fprintf (f, 'm4_define(`GB_copy_scalar_to_c'', `#define GB_COPY_scalar_to_C(Cx,pC,cwork) Cx [pC] = cwork'')\n') ;

% to copy an entry from A to C (no typecasting)
fprintf (f, 'm4_define(`GB_copy_aij_to_c'', `#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork) Cx [pC] = (A_iso) ? cwork : Ax [pA]'')\n') ;

% to copy an entry from A into a cwork scalar
fprintf (f, 'm4_define(`GB_copy_aij_to_cwork'', `#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) cwork = Ax [A_iso ? 0 : (pA)]'')\n') ;

% mask macro
if (isequal (xtype, 'GxB_FC32_t') || isequal (xtype, 'GxB_FC64_t'))
    asize = sprintf ('sizeof (%s)', xtype) ;
    fprintf (f, 'm4_define(`GB_ax_mask'', `#define GB_AX_MASK(Ax,pA,asize) GB_MCAST (Ax, pA, %s)'')\n', asize) ;
else
    fprintf (f, 'm4_define(`GB_ax_mask'', `#define GB_AX_MASK(Ax,pA,asize) (Ax [pA] != 0)'')\n') ;
end

% create the disable flag
disable = sprintf ('defined(GxB_NO_%s)', upper (fname)) ;
fprintf (f, 'm4_define(`GB_disable'', `#if (%s)\n#define GB_DISABLE 1\n#else\n#define GB_DISABLE 0\n#endif\n'')\n', disable) ;
fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

% construct the *.c file
cmd = sprintf ('cat control.m4 Generator/GB_as.c | m4 -P | awk -f codegen_blank.awk > FactoryKernels/GB_as__%s.c', fname) ;
system (cmd) ;

% append to the *.h file
system ('cat control.m4 Generator/GB_as.h | m4 -P | awk -f codegen_blank.awk | grep -v SPDX >> FactoryKernels/GB_as__include.h') ;

delete ('control.m4') ;

