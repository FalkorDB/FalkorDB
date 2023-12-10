function jit_reset
%JIT_RESET turn off the JIT and then set it back to its original state

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

c = GB_mex_jit_control ;
GB_mex_finalize ;
GB_mex_jit_control (c) ;


