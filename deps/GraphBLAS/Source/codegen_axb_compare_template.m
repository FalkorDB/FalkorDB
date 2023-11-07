function codegen_axb_compare_template (multop, bmult, mult)
%CODEGEN_AXB_COMPARE_TEMPLATE create a function for a semiring with a TxT -> bool multiplier

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n%-7s', multop) ;

% lor monoid
update = 'warg |= targ' ;
addfunc = 'zarg = xarg || yarg' ;
codegen_axb_method ('lor', multop, update, addfunc, bmult, 'bool', 'bool'    , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'int8_t'  , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'uint8_t' , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'int16_t' , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'uint16_t', 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'int32_t' , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'uint32_t', 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'int64_t' , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'uint64_t', 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'float'   , 'false', 'true') ;
codegen_axb_method ('lor', multop, update, addfunc,  mult, 'bool', 'double'  , 'false', 'true') ;

% any monoid
update = 'warg = targ' ;
addfunc = 'zarg = yarg' ;
codegen_axb_method ('any', multop, update, addfunc, bmult, 'bool', 'bool'    , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'int8_t'  , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'uint8_t' , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'int16_t' , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'uint16_t', 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'int32_t' , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'uint32_t', 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'int64_t' , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'uint64_t', 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'float'   , 'false', '(any value)') ;
codegen_axb_method ('any', multop, update, addfunc,  mult, 'bool', 'double'  , 'false', '(any value)') ;

% land monoid
update = 'warg &= targ' ;
addfunc = 'zarg = xarg && yarg' ;
codegen_axb_method ('land', multop, update, addfunc, bmult, 'bool', 'bool'    , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'int8_t'  , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'uint8_t' , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'int16_t' , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'uint16_t', 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'int32_t' , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'uint32_t', 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'int64_t' , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'uint64_t', 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'float'   , 'true', 'false') ;
codegen_axb_method ('land', multop, update, addfunc,  mult, 'bool', 'double'  , 'true', 'false') ;

% lxor monoid
update = 'warg ^= targ' ;
addfunc = 'zarg = (xarg != yarg)' ;
codegen_axb_method ('lxor', multop, update, addfunc, bmult, 'bool', 'bool'    , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'int8_t'  , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'uint8_t' , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'int16_t' , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'uint16_t', 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'int32_t' , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'uint32_t', 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'int64_t' , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'uint64_t', 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'float'   , 'false', [ ]) ;
codegen_axb_method ('lxor', multop, update, addfunc,  mult, 'bool', 'double'  , 'false', [ ]) ;

% eq (lxnor) monoid.  Cannot be done with OpenMP atomic
update = 'warg = (warg == targ)' ;
addfunc = 'zarg = (xarg == yarg)' ;
codegen_axb_method ('eq', multop, update, addfunc, bmult, 'bool', 'bool'    , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'int8_t'  , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'uint8_t' , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'int16_t' , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'uint16_t', 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'int32_t' , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'uint32_t', 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'int64_t' , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'uint64_t', 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'float'   , 'true', [ ]) ;
codegen_axb_method ('eq', multop, update, addfunc,  mult, 'bool', 'double'  , 'true', [ ]) ;

