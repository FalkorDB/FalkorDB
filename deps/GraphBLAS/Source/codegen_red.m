function codegen_red
%CODEGEN_RED create functions for all reduction operators
%
% This function creates all files of the form GB_red__*.c, GB_bld__*.c
% and the include files GB_red__include.h and GB_bld__include.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\nreduction operators:\n') ;

fh = fopen ('FactoryKernels/GB_red__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_red__include.h: definitions for GB_red__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_red.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

fh = fopen ('FactoryKernels/GB_bld__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_bld__include.h: definitions for GB_bld__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_bld.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

%-------------------------------------------------------------------------------
% the monoid: MIN, MAX, PLUS, TIMES, ANY, OR, AND, XOR, EQ
%-------------------------------------------------------------------------------

% Note that the min and max monoids are written to obtain the correct
% NaN behavior for float and double.  Compares with NaN are always false.
% zarg is the accumulator.  If zarg is not NaN and the compare is false,
% zarg is not modified and the value of yarg is properly ignored.  Thus if zarg
% is not NaN but yarg is NaN, then yarg is ignored.  If zarg is NaN, the
% condition becomes true and zarg is replaced with yarg.

% Panel sizes are optimal for gcc 8.3, on a MacBook.  They are probably fine
% for other architectures and compilers, too, but they haven't been tuned
% except for gcc 8.3 on a Mac.

% MIN: 10 monoids:  name      op   type        identity      terminal   panel
% (all but bool and complex)
fprintf ('\nmin    ') ;
op {1} = 'if (yarg < zarg) { zarg = yarg ; }' ;
op {2} = 'zarg = GB_IMIN (xarg, yarg)' ;
codegen_red_method ('min',    op, 'int8_t'  , 'INT8_MAX'  , 'INT8_MIN'  , 16) ;
codegen_red_method ('min',    op, 'int16_t' , 'INT16_MAX' , 'INT16_MIN' , 16) ;
codegen_red_method ('min',    op, 'int32_t' , 'INT32_MAX' , 'INT32_MIN' , 16) ;
codegen_red_method ('min',    op, 'int64_t' , 'INT64_MAX' , 'INT64_MIN' , 16) ;
codegen_red_method ('min',    op, 'uint8_t' , 'UINT8_MAX' , '0'         , 16) ;
codegen_red_method ('min',    op, 'uint16_t', 'UINT16_MAX', '0'         , 16) ;
codegen_red_method ('min',    op, 'uint32_t', 'UINT32_MAX', '0'         , 16) ;
codegen_red_method ('min',    op, 'uint64_t', 'UINT64_MAX', '0'         , 16) ;
op {1} = 'if ((yarg < zarg) || (zarg != zarg)) { zarg = yarg ; }' ;
op {2} = 'zarg = fminf (xarg, yarg)' ;
codegen_red_method ('min',    op, 'float'   , 'INFINITY' , [ ] , 16) ;
op {2} = 'zarg = fmin (xarg, yarg)' ;
codegen_red_method ('min',    op, 'double'  , '((double) INFINITY)', [ ], 16) ;

% MAX: 10 monoids (all but bool and complex)
fprintf ('\nmax    ') ;
op {1} = 'if (yarg > zarg) { zarg = yarg ; }' ;
op {2} = 'zarg = GB_IMAX (xarg, yarg)' ;
codegen_red_method ('max',    op, 'int8_t'  , 'INT8_MIN'  , 'INT8_MAX'  , 16) ;
codegen_red_method ('max',    op, 'int16_t' , 'INT16_MIN' , 'INT16_MAX' , 16) ;
codegen_red_method ('max',    op, 'int32_t' , 'INT32_MIN' , 'INT32_MAX' , 16) ;
codegen_red_method ('max',    op, 'int64_t' , 'INT64_MIN' , 'INT64_MAX' , 16) ;
codegen_red_method ('max',    op, 'uint8_t' , '0'         , 'UINT8_MAX' , 16) ;
codegen_red_method ('max',    op, 'uint16_t', '0'         , 'UINT16_MAX', 16) ;
codegen_red_method ('max',    op, 'uint32_t', '0'         , 'UINT32_MAX', 16) ;
codegen_red_method ('max',    op, 'uint64_t', '0'         , 'UINT64_MAX', 16) ;
op {1} = 'if ((yarg > zarg) || (zarg != zarg)) { zarg = yarg ; }' ;
op {2} = 'zarg = fmaxf (xarg, yarg)' ;
codegen_red_method ('max',    op, 'float'   , '(-INFINITY)' , [ ] , 16) ;
op {2} = 'zarg = fmax (xarg, yarg)' ;
codegen_red_method ('max',    op, 'double'  , '((double) -INFINITY)', [ ], 16) ;

% ANY: 13 monoids (including bool and complex)
fprintf ('\nany    ') ;
op {1} = 'zarg = yarg' ;
op {2} = 'zarg = yarg' ;
codegen_red_method ('any' ,   op, 'bool'      , '0') ;
codegen_red_method ('any',    op, 'int8_t'    , '0') ;
codegen_red_method ('any',    op, 'int16_t'   , '0') ;
codegen_red_method ('any',    op, 'int32_t'   , '0') ;
codegen_red_method ('any',    op, 'int64_t'   , '0') ;
codegen_red_method ('any',    op, 'uint8_t'   , '0') ;
codegen_red_method ('any',    op, 'uint16_t'  , '0') ;
codegen_red_method ('any',    op, 'uint32_t'  , '0') ;
codegen_red_method ('any',    op, 'uint64_t'  , '0') ;
codegen_red_method ('any',    op, 'float'     , '0') ;
codegen_red_method ('any',    op, 'double'    , '0') ;
codegen_red_method ('any',    op, 'GxB_FC32_t', 'GxB_CMPLXF(0,0)') ;
codegen_red_method ('any',    op, 'GxB_FC64_t', 'GxB_CMPLX(0,0)') ;

% PLUS: 12 monoids (all but bool)
fprintf ('\nplus   ') ;
op {1} = 'zarg += yarg' ;
op {2} = 'zarg = xarg + yarg' ;
codegen_red_method ('plus',   op, 'int8_t'    , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'int16_t'   , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'int32_t'   , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'int64_t'   , '0'              , [ ], 32) ;
codegen_red_method ('plus',   op, 'uint8_t'   , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'uint16_t'  , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'uint32_t'  , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'uint64_t'  , '0'              , [ ], 32) ;
codegen_red_method ('plus',   op, 'float'     , '0'              , [ ], 64) ;
codegen_red_method ('plus',   op, 'double'    , '0'              , [ ], 32) ;
op {1} = 'zarg = GB_FC32_add (zarg, yarg)' ;
op {2} = 'zarg = GB_FC32_add (xarg, yarg)' ;
codegen_red_method ('plus',   op, 'GxB_FC32_t', 'GxB_CMPLXF(0,0)', [ ], 32) ;
op {1} = 'zarg = GB_FC64_add (zarg, yarg)' ;
op {2} = 'zarg = GB_FC64_add (xarg, yarg)' ;
codegen_red_method ('plus',   op, 'GxB_FC64_t', 'GxB_CMPLX(0,0)' , [ ], 16) ;

% TIMES: 12 monoids (all but bool)
fprintf ('\ntimes  ') ;
op {1} = 'zarg *= yarg' ;
op {2} = 'zarg = xarg * yarg' ;
codegen_red_method ('times',  op, 'int8_t'    , '1'              , '0', 64) ;
codegen_red_method ('times',  op, 'int16_t'   , '1'              , '0', 64) ;
codegen_red_method ('times',  op, 'int32_t'   , '1'              , '0', 64) ;
codegen_red_method ('times',  op, 'int64_t'   , '1'              , '0', 16) ;
codegen_red_method ('times',  op, 'uint8_t'   , '1'              , '0', 64) ;
codegen_red_method ('times',  op, 'uint16_t'  , '1'              , '0', 64) ;
codegen_red_method ('times',  op, 'uint32_t'  , '1'              , '0', 64) ;
codegen_red_method ('times',  op, 'uint64_t'  , '1'              , '0', 16) ;
codegen_red_method ('times',  op, 'float'     , '1'              , [ ], 64) ;
codegen_red_method ('times',  op, 'double'    , '1'              , [ ], 32) ;
op {1} = 'zarg = GB_FC32_mul (zarg, yarg)' ;
op {2} = 'zarg = GB_FC32_mul (xarg, yarg)' ;
codegen_red_method ('times',  op, 'GxB_FC32_t', 'GxB_CMPLXF(1,0)', [ ], 32) ;
op {1} = 'zarg = GB_FC64_mul (zarg, yarg)' ;
op {2} = 'zarg = GB_FC64_mul (xarg, yarg)' ;
codegen_red_method ('times',  op, 'GxB_FC64_t', 'GxB_CMPLX(1,0)' , [ ], 16) ;

% 4 boolean monoids
fprintf ('\nlor    ') ;
op {1} = 'zarg = (zarg || yarg)' ;
op {2} = 'zarg = (xarg || yarg)' ;
codegen_red_method ('lor'  ,  op, 'bool','false', 'true' ,8);
fprintf ('\nland   ') ;
op {1} = 'zarg = (zarg && yarg)' ;
op {2} = 'zarg = (xarg && yarg)' ;
codegen_red_method ('land' ,  op, 'bool','true' , 'false',8);
fprintf ('\nlxor   ') ;
op {1} = 'zarg = (zarg != yarg)' ;
op {2} = 'zarg = (xarg != yarg)' ;
codegen_red_method ('lxor' ,  op, 'bool','false', [ ]    ,8);
fprintf ('\neq     ') ;
op {1} = 'zarg = (zarg == yarg)' ;
op {2} = 'zarg = (xarg == yarg)' ;
codegen_red_method ('eq'   ,  op, 'bool','true' , [ ]    ,8);
fprintf ('\nany    ') ;
op {1} = 'zarg = yarg' ;
op {2} = 'zarg = yarg' ;
codegen_red_method ('any'  ,  op, 'bool','false') ;

%-------------------------------------------------------------------------------
% FIRST and SECOND (not monoids; used for GB_bld__first,second_type)
%-------------------------------------------------------------------------------

% FIRST: 13 ops:    name      op           type        identity terminal panel
fprintf ('\nfirst  ') ;
op {1} = '' ;
op {2} = 'zarg = xarg' ;
codegen_red_method ('first',  op,  'bool'      , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'int8_t'    , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'int16_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'int32_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'int64_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'uint8_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'uint16_t'  , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'uint32_t'  , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'uint64_t'  , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'float'     , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'double'    , [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'GxB_FC32_t', [ ]  , [ ],   1  ) ;
codegen_red_method ('first',  op,  'GxB_FC64_t', [ ]  , [ ],   1  ) ;

% SECOND: 13 ops    name      op           type        identity terminal panel
fprintf ('\nsecond ') ;
op {1} = 'zarg = yarg' ;
op {2} = 'zarg = yarg' ;
codegen_red_method ('second', op,  'bool'      , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'int8_t'    , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'int16_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'int32_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'int64_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'uint8_t'   , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'uint16_t'  , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'uint32_t'  , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'uint64_t'  , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'float'     , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'double'    , [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'GxB_FC32_t', [ ]  , [ ],   1  ) ;
codegen_red_method ('second', op, 'GxB_FC64_t', [ ]  , [ ],   1  ) ;

fprintf ('\n') ;

