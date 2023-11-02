function codegen_axb_template (multop, bmult, imult, fmult, dmult, fcmult, dcmult, no_min_max_any_times_monoids)
%CODEGEN_AXB_TEMPLATE create a function for a semiring with a TxT->T multiplier

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n%-7s', multop) ;
if (nargin < 8)
    no_min_max_any_times_monoids = false ;
end

is_pair = isequal (multop, 'pair') ;
% the any_pair_iso semiring
if (is_pair)
    codegen_axb_method ('any', 'pair') ;
end

plusinf32 = 'INFINITY' ;
neginf32  = '(-INFINITY)' ;
plusinf64 = '((double) INFINITY)' ;
neginf64  = '((double) -INFINITY)' ;

% MIN monoid: integer types are terminal, float and double are not.
if (~no_min_max_any_times_monoids)
    update = 'warg = GB_IMIN (warg, targ)' ;
    addfunc = 'zarg = GB_IMIN (xarg, yarg)' ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'int8_t'  , 'int8_t'  , 'INT8_MAX'  , 'INT8_MIN'  ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'int16_t' , 'int16_t' , 'INT16_MAX' , 'INT16_MIN' ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'int32_t' , 'int32_t' , 'INT32_MAX' , 'INT32_MIN' ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'int64_t' , 'int64_t' , 'INT64_MAX' , 'INT64_MIN' ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'uint8_t' , 'uint8_t' , 'UINT8_MAX' , '0'         ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'uint16_t', 'uint16_t', 'UINT16_MAX', '0'         ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'uint32_t', 'uint32_t', 'UINT32_MAX', '0'         ) ;
    codegen_axb_method ('min', multop, update, addfunc, imult, 'uint64_t', 'uint64_t', 'UINT64_MAX', '0'         ) ;
    update = 'warg = fminf (warg, targ)' ;
    addfunc = 'zarg = fminf (xarg, yarg)' ;
    codegen_axb_method ('min', multop, update, addfunc, fmult, 'float'   , 'float'   , plusinf32   , [ ]         ) ;
    update = 'warg = fmin (warg, targ)' ;
    addfunc = 'zarg = fmin (xarg, yarg)' ;
    codegen_axb_method ('min', multop, update, addfunc, dmult, 'double'  , 'double'  , plusinf64   , [ ]         ) ;
end

% MAX monoid: integer types are terminal, float and double are not.
if (~no_min_max_any_times_monoids)
    update = 'warg = GB_IMAX (warg, targ)' ;
    addfunc = 'zarg = GB_IMAX (xarg, yarg)' ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'int8_t'  , 'int8_t'  , 'INT8_MIN'  , 'INT8_MAX'  ) ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'int16_t' , 'int16_t' , 'INT16_MIN' , 'INT16_MAX' ) ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'int32_t' , 'int32_t' , 'INT32_MIN' , 'INT32_MAX' ) ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'int64_t' , 'int64_t' , 'INT64_MIN' , 'INT64_MAX' ) ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'uint8_t' , 'uint8_t' , '0'         , 'UINT8_MAX' ) ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'uint16_t', 'uint16_t', '0'         , 'UINT16_MAX') ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'uint32_t', 'uint32_t', '0'         , 'UINT32_MAX') ;
    codegen_axb_method ('max', multop, update, addfunc, imult, 'uint64_t', 'uint64_t', '0'         , 'UINT64_MAX') ;
    % floating-point MAX must use unsigned integer puns for compare-and-swap
    update = 'warg = fmaxf (warg, targ)' ;
    addfunc = 'zarg = fmaxf (xarg, yarg)' ;
    codegen_axb_method ('max', multop, update, addfunc, fmult, 'float'   , 'float'   , neginf32    , [ ]         ) ;
    update = 'warg = fmax (warg, targ)' ;
    addfunc = 'zarg = fmax (xarg, yarg)' ;
    codegen_axb_method ('max', multop, update, addfunc, dmult, 'double'  , 'double'  , neginf64    , [ ]         ) ;
end

% ANY monoid: all are terminal.
if (~no_min_max_any_times_monoids && ~is_pair)
    update = 'warg = targ' ;
    addfunc = 'zarg = yarg' ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'int8_t'  , 'int8_t'  , '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'int16_t' , 'int16_t' , '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'int32_t' , 'int32_t' , '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'int64_t' , 'int64_t' , '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'uint8_t' , 'uint8_t' , '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'uint16_t', 'uint16_t', '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'uint32_t', 'uint32_t', '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, imult, 'uint64_t', 'uint64_t', '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, fmult, 'float'   , 'float'   , '0' , '(any value)') ;
    codegen_axb_method ('any', multop, update, addfunc, dmult, 'double'  , 'double'  , '0' , '(any value)') ;
    % complex case:
    id = 'GxB_CMPLXF(0,0)' ;
    codegen_axb_method ('any', multop, update, addfunc, fcmult, 'GxB_FC32_t', 'GxB_FC32_t', id, '(any value)') ;
    id = 'GxB_CMPLX(0,0)' ;
    codegen_axb_method ('any', multop, update, addfunc, dcmult, 'GxB_FC64_t', 'GxB_FC64_t', id, '(any value)') ;
end

% PLUS monoid: none are terminal.
update = 'warg += targ' ;
addfunc = 'zarg = xarg + yarg' ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'int8_t'  , 'int8_t'  , '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'uint8_t' , 'uint8_t' , '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'int16_t' , 'int16_t' , '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'uint16_t', 'uint16_t', '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'int32_t' , 'int32_t' , '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'uint32_t', 'uint32_t', '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'int64_t' , 'int64_t' , '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, imult, 'uint64_t', 'uint64_t', '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, fmult, 'float'   , 'float'   , '0', [ ]) ;
codegen_axb_method ('plus', multop, update, addfunc, dmult, 'double'  , 'double'  , '0', [ ]) ;
update = 'warg = GB_FC32_add (warg, targ)' ;
addfunc = 'zarg = GB_FC32_add (xarg, yarg)' ;
id = 'GxB_CMPLXF(0,0)' ;
codegen_axb_method ('plus', multop, update, addfunc, fcmult, 'GxB_FC32_t', 'GxB_FC32_t', id, [ ]) ;
update = 'warg = GB_FC64_add (warg, targ)' ;
addfunc = 'zarg = GB_FC64_add (xarg, yarg)' ;
id = 'GxB_CMPLX(0,0)' ;
codegen_axb_method ('plus', multop, update, addfunc, dcmult, 'GxB_FC64_t', 'GxB_FC64_t', id, [ ]) ;

% TIMES monoid: integers are terminal, float and double are not.
if (~no_min_max_any_times_monoids)
    update = 'warg *= targ' ;
    addfunc = 'zarg = xarg * yarg' ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'int8_t'  , 'int8_t'  , '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'uint8_t' , 'uint8_t' , '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'int16_t' , 'int16_t' , '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'uint16_t', 'uint16_t', '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'int32_t' , 'int32_t' , '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'uint32_t', 'uint32_t', '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'int64_t' , 'int64_t' , '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, imult, 'uint64_t', 'uint64_t', '1', '0') ;
    codegen_axb_method ('times', multop, update, addfunc, fmult, 'float'   , 'float'   , '1', [ ]) ;
    codegen_axb_method ('times', multop, update, addfunc, dmult, 'double'  , 'double'  , '1', [ ]) ;
    update = 'warg = GB_FC32_mul (warg, targ)' ;
    addfunc = 'zarg = GB_FC32_mul (xarg, yarg)' ;
    id = 'GxB_CMPLXF(1,0)' ;
    codegen_axb_method ('times', multop, update, addfunc, fcmult, 'GxB_FC32_t', 'GxB_FC32_t', id, [ ]) ;
    update = 'warg = GB_FC64_mul (warg, targ)' ;
    addfunc = 'zarg = GB_FC64_mul (xarg, yarg)' ;
    id = 'GxB_CMPLX(1,0)' ;
    codegen_axb_method ('times', multop, update, addfunc, dcmult, 'GxB_FC64_t', 'GxB_FC64_t', id, [ ]) ;
end

% boolean monoids: LOR, LAND are terminal; LXOR, EQ are not.
addfunc = 'zarg = xarg || yarg' ;
codegen_axb_method ('lor',  multop, 'warg |= targ', addfunc, bmult, 'bool', 'bool', 'false', 'true' ) ;
addfunc = 'zarg = xarg && yarg' ;
codegen_axb_method ('land', multop, 'warg &= targ', addfunc, bmult, 'bool', 'bool', 'true' , 'false') ;
addfunc = 'zarg = xarg != yarg' ;
codegen_axb_method ('lxor', multop, 'warg ^= targ', addfunc, bmult, 'bool', 'bool', 'false', [ ]    ) ;
if (~is_pair)
    addfunc = 'zarg = yarg' ;
    codegen_axb_method ('any', multop, 'warg = targ' , addfunc , bmult, 'bool', 'bool', '0'    , '(any value)') ;
end
update = 'warg = (warg == targ)' ;
addfunc = 'zarg = xarg == yarg' ;
codegen_axb_method ('eq',   multop, update,      addfunc, bmult, 'bool', 'bool', 'true' , [ ]    ) ;

