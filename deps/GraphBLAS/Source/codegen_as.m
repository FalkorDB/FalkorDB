function codegen_as
%CODEGEN_AS create functions for assign/subassign methods with no accum
%
% This function creates all files of the form GB_as__*.[ch], including 13
% functions (GB_as__*.c) and one include file, GB_as__include.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\nsubassign/assign with no accum:\n') ;

fh = fopen ('FactoryKernels/GB_as__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_as__include.h: definitions for GB_as__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_as.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

codegen_as_template ('bool') ;
codegen_as_template ('int8_t') ;
codegen_as_template ('int16_t') ;
codegen_as_template ('int32_t') ;
codegen_as_template ('int64_t') ;
codegen_as_template ('uint8_t') ;
codegen_as_template ('uint16_t') ;
codegen_as_template ('uint32_t') ;
codegen_as_template ('uint64_t') ;
codegen_as_template ('float') ;
codegen_as_template ('double') ;
codegen_as_template ('GxB_FC32_t') ;
codegen_as_template ('GxB_FC64_t') ;

fprintf ('\n') ;

