function codegen_aop
%CODEGEN_ASSIGN create functions for all binary operators for assign/subassign
%
% This function creates all files of the form GB_aop__*.[ch], including 260
% functions (GB_aop__*.c) and one include file, GB_aop__include.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\nassign/subassign with accum operator:\n') ;

fh = fopen ('FactoryKernels/GB_aop__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_aop__include.h: definitions for GB_aop__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_aop.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

% The ANY operator is not used as a binary operator in the generated functions.
% It can be used as the binary op in eWiseAdd, eWiseMult, etc, but has been
% renamed to SECOND before calling the generated function.

codegen_aop_template ('first',        ...
    'xarg',                             ... % bool
    'xarg',                             ... % int, uint
    'xarg',                             ... % float
    'xarg',                             ... % double
    'xarg',                             ... % GxB_FC32_t
    'xarg') ;                           ... % GxB_FC64_t

codegen_aop_template ('second',       ...
    'yarg',                             ... % bool
    'yarg',                             ... % int, uint
    'yarg',                             ... % float
    'yarg',                             ... % double
    'yarg',                             ... % GxB_FC32_t
    'yarg') ;                           ... % GxB_FC64_t

codegen_aop_template ('pair',         ...
    '1',                                ... % bool
    '1',                                ... % int, uint
    '1',                                ... % float
    '1',                                ... % double
    'GxB_CMPLXF(1,0)',                  ... % GxB_FC32_t
    'GxB_CMPLX(1,0)') ;                 ... % GxB_FC64_t

codegen_aop_template ('min',          ...
    [ ],                                ... % bool
    'GB_IMIN (xarg, yarg)',             ... % int, uint
    'fminf (xarg, yarg)',               ... % float
    'fmin (xarg, yarg)',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('max',          ...
    [ ],                                ... % bool
    'GB_IMAX (xarg, yarg)',             ... % int, uint
    'fmaxf (xarg, yarg)',               ... % float
    'fmax (xarg, yarg)',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('plus',         ...
    [ ],                                ... % bool
    '(xarg) + (yarg)',                  ... % int, uint
    '(xarg) + (yarg)',                  ... % float
    '(xarg) + (yarg)',                  ... % double
    'GB_FC32_add (xarg, yarg)',         ... % GxB_FC32_t
    'GB_FC64_add (xarg, yarg)') ;       ... % GxB_FC64_t

codegen_aop_template ('minus',        ...
    [ ],                                ... % bool
    '(xarg) - (yarg)',                  ... % int, uint
    '(xarg) - (yarg)',                  ... % float
    '(xarg) - (yarg)',                  ... % double
    'GB_FC32_minus (xarg, yarg)',       ... % GxB_FC32_t
    'GB_FC64_minus (xarg, yarg)') ;     ... % GxB_FC64_t

codegen_aop_template ('rminus',       ...
    [ ],                                ... % bool
    '(yarg) - (xarg)',                  ... % int, uint
    '(yarg) - (xarg)',                  ... % float
    '(yarg) - (xarg)',                  ... % double
    'GB_FC32_minus (yarg, xarg)',       ... % GxB_FC32_t
    'GB_FC64_minus (yarg, xarg)') ;     ... % GxB_FC64_t

codegen_aop_template ('times',        ...
    [ ],                                ... % bool
    '(xarg) * (yarg)',                  ... % int, uint
    '(xarg) * (yarg)',                  ... % float
    '(xarg) * (yarg)',                  ... % double
    'GB_FC32_mul (xarg, yarg)',         ... % GxB_FC32_t
    'GB_FC64_mul (xarg, yarg)') ;       ... % GxB_FC64_t

codegen_aop_template ('div',          ...
    [ ],                                ... % bool
    'GB_idiv (xarg, yarg)',             ... % int, uint
    '(xarg) / (yarg)',                  ... % float
    '(xarg) / (yarg)',                  ... % double
    'GB_FC32_div (xarg, yarg)',         ... % GxB_FC32_t
    'GB_FC64_div (xarg, yarg)') ;       ... % GxB_FC64_t

codegen_aop_template ('rdiv',         ...
    [ ],                                ... % bool
    'GB_idiv (yarg, xarg)',             ... % int, uint
    '(yarg) / (xarg)',                  ... % float
    '(yarg) / (xarg)',                  ... % double
    'GB_FC32_div (yarg, xarg)',         ... % GxB_FC32_t
    'GB_FC64_div (yarg, xarg)') ;       ... % GxB_FC64_t

codegen_aop_template ('iseq',         ...
    [ ],                                ... % bool
    '((xarg) == (yarg))',               ... % int, uint
    '((xarg) == (yarg))',               ... % float
    '((xarg) == (yarg))',               ... % double
    'GB_FC32_iseq (xarg, yarg)',        ... % GxB_FC32_t
    'GB_FC64_iseq (xarg, yarg)') ;      ... % GxB_FC64_t

codegen_aop_template ('isne',         ...
    [ ],                                ... % bool
    '((xarg) != (yarg))',               ... % int, uint
    '((xarg) != (yarg))',               ... % float
    '((xarg) != (yarg))',               ... % double
    'GB_FC32_isne (xarg, yarg)',        ... % GxB_FC32_t
    'GB_FC64_isne (xarg, yarg)') ;      ... % GxB_FC64_t

codegen_aop_template ('isgt',         ...
    [ ],                                ... % bool
    '((xarg) > (yarg))',                ... % int, uint
    '((xarg) > (yarg))',                ... % float
    '((xarg) > (yarg))',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('islt',         ...
    [ ],                                ... % bool
    '((xarg) < (yarg))',                ... % int, uint
    '((xarg) < (yarg))',                ... % float
    '((xarg) < (yarg))',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('isge',         ...
    [ ],                                ... % bool
    '((xarg) >= (yarg))',               ... % int, uint
    '((xarg) >= (yarg))',               ... % float
    '((xarg) >= (yarg))',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('isle',         ...
    [ ],                                ... % bool
    '((xarg) <= (yarg))',               ... % int, uint
    '((xarg) <= (yarg))',               ... % float
    '((xarg) <= (yarg))',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('eq',           ...
    '((xarg) == (yarg))',               ... % bool
    '((xarg) == (yarg))',               ... % int, uint
    '((xarg) == (yarg))',               ... % float
    '((xarg) == (yarg))',               ... % double
    'GB_FC32_eq (xarg, yarg)',          ... % GxB_FC32_t
    'GB_FC64_eq (xarg, yarg)') ;        ... % GxB_FC64_t

codegen_aop_template ('ne',           ...
    [ ],                                ... % bool
    '((xarg) != (yarg))',               ... % int, uint
    '((xarg) != (yarg))',               ... % float
    '((xarg) != (yarg))',               ... % double
    'GB_FC32_ne (xarg, yarg)',          ... % GxB_FC32_t
    'GB_FC64_ne (xarg, yarg)') ;        ... % GxB_FC64_t

codegen_aop_template ('gt',           ...
    '((xarg) > (yarg))',                ... % bool
    '((xarg) > (yarg))',                ... % int, uint
    '((xarg) > (yarg))',                ... % float
    '((xarg) > (yarg))',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('lt',           ...
    '((xarg) < (yarg))',                ... % bool
    '((xarg) < (yarg))',                ... % int, uint
    '((xarg) < (yarg))',                ... % float
    '((xarg) < (yarg))',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('ge',           ...
    '((xarg) >= (yarg))',               ... % bool
    '((xarg) >= (yarg))',               ... % int, uint
    '((xarg) >= (yarg))',               ... % float
    '((xarg) >= (yarg))',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('le',           ...
    '((xarg) <= (yarg))',               ... % bool
    '((xarg) <= (yarg))',               ... % int, uint
    '((xarg) <= (yarg))',               ... % float
    '((xarg) <= (yarg))',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('lor',          ...
    '((xarg) || (yarg))',               ... % bool
    '(((xarg) != 0) || ((yarg) != 0))', ... % int, uint
    '(((xarg) != 0) || ((yarg) != 0))', ... % float
    '(((xarg) != 0) || ((yarg) != 0))', ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('land',         ...
    '((xarg) && (yarg))',               ... % bool
    '(((xarg) != 0) && ((yarg) != 0))', ... % int, uint
    '(((xarg) != 0) && ((yarg) != 0))', ... % float
    '(((xarg) != 0) && ((yarg) != 0))', ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('lxor',         ...
    '((xarg) != (yarg))',               ... % bool
    '(((xarg) != 0) != ((yarg) != 0))', ... % int, uint
    '(((xarg) != 0) != ((yarg) != 0))', ... % float
    '(((xarg) != 0) != ((yarg) != 0))', ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('atan2',        ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'atan2f (xarg, yarg)',              ... % float
    'atan2 (xarg, yarg)',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('hypot',        ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'hypotf (xarg, yarg)',              ... % float
    'hypot (xarg, yarg)',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('fmod',         ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'fmodf (xarg, yarg)',               ... % float
    'fmod (xarg, yarg)',                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('remainder',    ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'remainderf (xarg, yarg)',          ... % float
    'remainder (xarg, yarg)',           ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('copysign',     ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'copysignf (xarg, yarg)',           ... % float
    'copysign (xarg, yarg)',            ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('ldexp',        ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'ldexpf (xarg, yarg)',              ... % float
    'ldexp (xarg, yarg)',               ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('cmplx',        ...
    [ ],                                ... % bool
    [ ],                                ... % int, uint
    'GJ_CMPLX32 (xarg, yarg)',          ... % float  (z is GxB_FC32_t)
    'GJ_CMPLX64 (xarg, yarg)',          ... % double (z is GxB_FC64_t)
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('bor',          ...
    [ ],                                ... % bool
    '((xarg) | (yarg))',                ... % int, uint
    [ ],                                ... % float
    [ ],                                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('band',         ...
    [ ],                                ... % bool
    '((xarg) & (yarg))',                ... % int, uint
    [ ],                                ... % float
    [ ],                                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('bxor',         ...
    [ ],                                ... % bool
    '((xarg) ^ (yarg))',                ... % int, uint
    [ ],                                ... % float
    [ ],                                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

codegen_aop_template ('bxnor',        ...
    [ ],                                ... % bool
    '~((xarg) ^ (yarg))',               ... % int, uint
    [ ],                                ... % float
    [ ],                                ... % double
    [ ],                                ... % GxB_FC32_t
    [ ]) ;                              ... % GxB_FC64_t

% bget
fprintf ('\nbget     ') ;
codegen_aop_method ('bget', 'GB_bitget_int8 (xarg, yarg)'  , 'int8_t'  ) ;
codegen_aop_method ('bget', 'GB_bitget_int16 (xarg, yarg)' , 'int16_t' ) ;
codegen_aop_method ('bget', 'GB_bitget_int32 (xarg, yarg)' , 'int32_t' ) ;
codegen_aop_method ('bget', 'GB_bitget_int64 (xarg, yarg)' , 'int64_t' ) ;
codegen_aop_method ('bget', 'GB_bitget_uint8 (xarg, yarg)' , 'uint8_t' ) ;
codegen_aop_method ('bget', 'GB_bitget_uint16 (xarg, yarg)', 'uint16_t') ;
codegen_aop_method ('bget', 'GB_bitget_uint32 (xarg, yarg)', 'uint32_t') ;
codegen_aop_method ('bget', 'GB_bitget_uint64 (xarg, yarg)', 'uint64_t') ;

% bset
fprintf ('\nbset     ') ;
codegen_aop_method ('bset', 'GB_bitset_int8 (xarg, yarg)'  , 'int8_t'  ) ;
codegen_aop_method ('bset', 'GB_bitset_int16 (xarg, yarg)' , 'int16_t' ) ;
codegen_aop_method ('bset', 'GB_bitset_int32 (xarg, yarg)' , 'int32_t' ) ;
codegen_aop_method ('bset', 'GB_bitset_int64 (xarg, yarg)' , 'int64_t' ) ;
codegen_aop_method ('bset', 'GB_bitset_uint8 (xarg, yarg)' , 'uint8_t' ) ;
codegen_aop_method ('bset', 'GB_bitset_uint16 (xarg, yarg)', 'uint16_t') ;
codegen_aop_method ('bset', 'GB_bitset_uint32 (xarg, yarg)', 'uint32_t') ;
codegen_aop_method ('bset', 'GB_bitset_uint64 (xarg, yarg)', 'uint64_t') ;

% bclr
fprintf ('\nbclr     ') ;
codegen_aop_method ('bclr', 'GB_bitclr_int8 (xarg, yarg)'  , 'int8_t'  ) ;
codegen_aop_method ('bclr', 'GB_bitclr_int16 (xarg, yarg)' , 'int16_t' ) ;
codegen_aop_method ('bclr', 'GB_bitclr_int32 (xarg, yarg)' , 'int32_t' ) ;
codegen_aop_method ('bclr', 'GB_bitclr_int64 (xarg, yarg)' , 'int64_t' ) ;
codegen_aop_method ('bclr', 'GB_bitclr_uint8 (xarg, yarg)' , 'uint8_t' ) ;
codegen_aop_method ('bclr', 'GB_bitclr_uint16 (xarg, yarg)', 'uint16_t') ;
codegen_aop_method ('bclr', 'GB_bitclr_uint32 (xarg, yarg)', 'uint32_t') ;
codegen_aop_method ('bclr', 'GB_bitclr_uint64 (xarg, yarg)', 'uint64_t') ;

% bshift
fprintf ('\nbshift   ') ;
codegen_aop_method ('bshift', 'GB_bitshift_int8 (xarg, yarg)'  , 'int8_t'  ) ;
codegen_aop_method ('bshift', 'GB_bitshift_int16 (xarg, yarg)' , 'int16_t' ) ;
codegen_aop_method ('bshift', 'GB_bitshift_int32 (xarg, yarg)' , 'int32_t' ) ;
codegen_aop_method ('bshift', 'GB_bitshift_int64 (xarg, yarg)' , 'int64_t' ) ;
codegen_aop_method ('bshift', 'GB_bitshift_uint8 (xarg, yarg)' , 'uint8_t' ) ;
codegen_aop_method ('bshift', 'GB_bitshift_uint16 (xarg, yarg)', 'uint16_t') ;
codegen_aop_method ('bshift', 'GB_bitshift_uint32 (xarg, yarg)', 'uint32_t') ;
codegen_aop_method ('bshift', 'GB_bitshift_uint64 (xarg, yarg)', 'uint64_t') ;

% pow
fprintf ('\npow      ') ;
codegen_aop_method ('pow', 'GB_pow_int8 (xarg, yarg)'  , 'int8_t'    ) ;
codegen_aop_method ('pow', 'GB_pow_int16 (xarg, yarg)' , 'int16_t'   ) ;
codegen_aop_method ('pow', 'GB_pow_int32 (xarg, yarg)' , 'int32_t'   ) ;
codegen_aop_method ('pow', 'GB_pow_int64 (xarg, yarg)' , 'int64_t'   ) ;
codegen_aop_method ('pow', 'GB_pow_uint8 (xarg, yarg)' , 'uint8_t'   ) ;
codegen_aop_method ('pow', 'GB_pow_uint16 (xarg, yarg)', 'uint16_t'  ) ;
codegen_aop_method ('pow', 'GB_pow_uint32 (xarg, yarg)', 'uint32_t'  ) ;
codegen_aop_method ('pow', 'GB_pow_uint64 (xarg, yarg)', 'uint64_t'  ) ;
codegen_aop_method ('pow', 'GB_powf (xarg, yarg)'      , 'float'     ) ;
codegen_aop_method ('pow', 'GB_pow (xarg, yarg)'       , 'double'    ) ;
codegen_aop_method ('pow', 'GB_FC32_pow (xarg, yarg)'  , 'GxB_FC32_t') ;
codegen_aop_method ('pow', 'GB_FC64_pow (xarg, yarg)'  , 'GxB_FC64_t') ;

fprintf ('\n') ;

