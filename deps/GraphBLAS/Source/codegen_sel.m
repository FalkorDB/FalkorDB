function codegen_sel
%CODEGEN_SEL create functions for all selection operators
%
% This function creates all files of the form GB_sel__*.c,
% and the include file GB_sel__include.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\nselection operators:\n') ;
addpath ('../Test') ;

fh = fopen ('FactoryKernels/GB_sel__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_sel__include.h: definitions for GB_sel__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_sel.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

% NONZOMBIE:         name         selector                     type
fprintf ('\nnonzombie  ') ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'bool'      ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'int8_t'    ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'int16_t'   ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'int32_t'   ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'int64_t'   ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'uint8_t'   ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'uint16_t'  ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'uint32_t'  ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'uint64_t'  ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'float'     ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'double'    ) ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'GxB_FC32_t') ;
codegen_sel_method ('nonzombie', 'bool keep = (i >= 0)', 'GxB_FC64_t') ;

% NE_THUNK           name         selector            type
fprintf ('\nne_thunk   ') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'int8_t'  ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'int16_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'int32_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'int64_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'uint8_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'uint16_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'uint32_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'uint64_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'float'   ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != y)', 'double'  ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = GB_FC32_ne (Ax [p], y)', 'GxB_FC32_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = GB_FC64_ne (Ax [p], y)', 'GxB_FC64_t') ;

% EQ_THUNK           name         selector            type
fprintf ('\neq_thunk   ') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'bool'    ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'int8_t'  ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'int16_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'int32_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'int64_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'uint8_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'uint16_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'uint32_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'uint64_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'float'   ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == y)', 'double'  ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = GB_FC32_eq (Ax [p], y)', 'GxB_FC32_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = GB_FC64_eq (Ax [p], y)', 'GxB_FC64_t') ;

% GT_THUNK           name         selector            type
fprintf ('\ngt_thunk   ') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'int8_t'  ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'int16_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'int32_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'int64_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'uint8_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'uint16_t') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'uint32_t') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'uint64_t') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'float'   ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > y)', 'double'  ) ;

% GE_THUNK           name         selector            type
fprintf ('\nge_thunk   ') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'int8_t'  ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'int16_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'int32_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'int64_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'uint8_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'uint16_t') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'uint32_t') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'uint64_t') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'float'   ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= y)', 'double'  ) ;

% LT_THUNK           name         selector            type
fprintf ('\nlt_thunk   ') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'int8_t'  ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'int16_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'int32_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'int64_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'uint8_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'uint16_t') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'uint32_t') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'uint64_t') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'float'   ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < y)', 'double'  ) ;

% LE_THUNK           name         selector            type
fprintf ('\nle_thunk   ') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'int8_t'  ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'int16_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'int32_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'int64_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'uint8_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'uint16_t') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'uint32_t') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'uint64_t') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'float'   ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= y)', 'double'  ) ;

fprintf ('\n') ;

