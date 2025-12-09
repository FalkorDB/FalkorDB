function codegen_axb_method (addop, multop, update, addfunc, mult, ztype, xytype, identity, terminal)
%CODEGEN_AXB_METHOD create a function to compute C=A*B over a semiring
%
% codegen_axb_method (addop, multop, update, addfunc, mult, ztype, xytype, identity, terminal)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin >= 5 && isempty (mult))
    return
end

f = fopen ('control.m4', 'w') ;
fprintf (f, 'm4_divert(-1)\n') ;

is_first  = false ;
is_second = false ;
is_pair   = false ;
is_positional = false ;
switch (multop)
    case { 'firsti', 'firsti1', 'firstj', 'firstj1', 'secondj', 'secondj1' }
        is_positional = true ;
    case { 'first' }
        is_first = true ;
    case { 'second' }
        is_second = true ;
    case { 'pair' }
        is_pair = true ;
end

is_any = isequal (addop, 'any') ;
is_max = isequal (addop, 'max') ;
is_min = isequal (addop, 'min') ;
is_eq  = isequal (addop, 'eq') ;
is_lxor = isequal (addop, 'lxor') ;
is_plus = isequal (addop, 'plus') ;
is_any_pair = is_any && is_pair ;

if (is_any_pair)
    % the any_pair_iso semiring does not access any numerical values
    update = ' ' ;
    addfunc = ' ' ;
    mult = ' ' ;
    ztype = 'iso' ;
    xytype = 'any type' ;
    identity = '(any value)' ;
    terminal = '(any value)' ;
    fprintf (f, 'm4_define(`GB_axb__include_h'', `#include "mxm/GB_AxB__include1.h"'')\n') ;
else
    fprintf (f, 'm4_define(`GB_axb__include_h'', `#include "FactoryKernels/GB_AxB__include2.h"'')\n') ;
end

is_integer = codegen_contains (ztype, 'int') ;

ztype_is_bool = isequal (ztype, 'bool') ;
ztype_is_real = ~codegen_contains (ztype, 'FC') ;
ztype_is_fp = isequal (ztype, 'float') || isequal (ztype, 'double') ;
is_any_complex = is_any && ~ztype_is_real ;
is_plus_pair_real = is_plus && is_pair && (is_integer || ztype_is_fp) ;
is_plus_times_fp = is_plus && isequal (multop, 'times') && ztype_is_fp ;

t_is_simple = is_pair || codegen_contains (multop, 'first') || codegen_contains (multop, 'second') ;
t_is_nonnan = isequal (multop (1:2), 'is') || (multop (1) == 'l') ;

ztype_atomic = '' ;

switch (ztype)

    case { 'iso' }
        ztype_is_float = false ;
        ztype_ignore_overflow = true ;
        ztype_nbits = 0 ;

    case { 'bool' }
        ztype_is_float = false ;
        ztype_ignore_overflow = false ;
        ztype_nbits = 8 ;

    case { 'int8_t', 'uint8_t' }
        ztype_is_float = false ;
        ztype_ignore_overflow = false ;
        ztype_nbits = 8 ;

    case { 'int16_t', 'uint16_t' }
        ztype_is_float = false ;
        ztype_ignore_overflow = false ;
        ztype_nbits = 16 ;

    case { 'int32_t', 'uint32_t' }
        ztype_is_float = false ;
        ztype_ignore_overflow = false ;
        ztype_nbits = 32 ;

    case { 'int64_t', 'uint64_t' }
        ztype_is_float = false ;
        ztype_ignore_overflow = true ;
        ztype_nbits = 64 ;

    case { 'float' }
        ztype_is_float = true ;
        ztype_ignore_overflow = true ;
        ztype_nbits = 32 ;

    case { 'double' }
        ztype_is_float = true ;
        ztype_ignore_overflow = true ;
        ztype_nbits = 64 ;

    case { 'GxB_FC32_t' }
        ztype_is_float = true ;
        ztype_ignore_overflow = true ;
        ztype_nbits = 64 ;
        ztype_atomic = 'uint64_t' ;

    case { 'GxB_FC64_t' }
        ztype_is_float = true ;
        ztype_ignore_overflow = true ;
        ztype_nbits = 128 ;

    otherwise
        error ('unknown type') ;
end

fprintf (f, 'm4_define(`GB_z_nbits'', `#define GB_Z_NBITS %d'')\n', ztype_nbits) ;

% atomic write and compare/exchange
has_atomic_write = false ;
if (ztype_nbits <= 64)
    % atomic write and compare/exchange can only be done on 64 bits or less
    has_atomic_write = true ;
    fprintf (f, 'm4_define(`GB_z_atomic_bits'', `#define GB_Z_ATOMIC_BITS %d'')\n', ztype_nbits) ;
else
    % no atomics for this data type
    fprintf (f, 'm4_define(`GB_z_atomic_bits'', `'')\n') ;
end

% atomic ztype for pun
if (isempty (ztype_atomic))
    % no need for atomic pun
    fprintf (f, 'm4_define(`GB_z_atomic_type'', `'')\n') ;
else
    % use the pun for this ztype
    fprintf (f, 'm4_define(`GB_z_atomic_type'', `#define GB_Z_ATOMIC_TYPE %s'')\n', ztype_atomic) ;
end

% atomic update
has_atomic_update = false ;
omp_atomic_version = 0 ;

if (isequal (addop, 'any'))

    % ANY monoid update can be done atomically via atomic write
    has_atomic_update = has_atomic_write ; 
    if (ztype_is_real)
        omp_atomic_version = 2 ;
    else
        % complex ANY monoid update; could be done with a single atomic write
        % for fc32 using an uint64_t pun, but this is not needed for the
        % pre-generated kernels.
        omp_atomic_version = 0 ;
    end

elseif (isequal (addop, 'land') || ...
        isequal (addop, 'lor')  || ...
        isequal (addop, 'lxor') || ...
        isequal (addop, 'band') || ...
        isequal (addop, 'bor')  || ...
        isequal (addop, 'bxor'))

    % atomic update but #pramga omp update requires OpenMP 4.0 or later
    has_atomic_update = true ;
    omp_atomic_version = 4 ;

elseif (isequal (addop, 'bxnor') || ...
        isequal (addop, 'lxnor') || ...
        isequal (addop, 'eq')    || ...
        isequal (addop, 'min')   || ...
        isequal (addop, 'max'))

    % atomic update but no #pramga omp update
    has_atomic_update = true ;
    omp_atomic_version = 0 ;

elseif (isequal (addop, 'plus'))

    % PLUS monoid can be done atomically for types, even double complex
    has_atomic_update = true ;
    omp_atomic_version = 2 ;

elseif (isequal (addop, 'times'))

    % TIMES monoid can be done atomically for real types and single complex
    if (ztype_is_real)
        % real types have an omp pragma
        has_atomic_update = true ;
        omp_atomic_version = 2 ;
    elseif (isequal (ztype, 'GxB_FC32_t'))
        % single complex can be done as compare-and-swap
        % but this fails with gcc on Power8
        % has_atomic_update = true ;
        has_atomic_update = false ;
        omp_atomic_version = 0 ;
    end

else

    addop
    error ('undefined monoid') ;

end

if (has_atomic_update)
    fprintf (f, 'm4_define(`GB_z_has_atomic_update'', `#define GB_Z_HAS_ATOMIC_UPDATE 1'')\n') ;
    if (omp_atomic_version == 4)
        fprintf (f, 'm4_define(`GB_z_has_omp_atomic_update'', `#define GB_Z_HAS_OMP_ATOMIC_UPDATE (!GB_COMPILER_MSC)'')\n') ;
    elseif (omp_atomic_version == 2)
        fprintf (f, 'm4_define(`GB_z_has_omp_atomic_update'', `#define GB_Z_HAS_OMP_ATOMIC_UPDATE 1'')\n') ;
    else
        fprintf (f, 'm4_define(`GB_z_has_omp_atomic_update'', `'')\n') ;
    end
else
    fprintf (f, 'm4_define(`GB_z_has_atomic_update'', `'')\n') ;
    fprintf (f, 'm4_define(`GB_z_has_omp_atomic_update'', `'')\n') ;
end

if (is_pair)
    % these semirings are renamed to any_pair, and not thus created
    if (isequal (addop, 'land') || isequal (addop, 'eq'   ) || ...
        isequal (addop, 'lor' ) || isequal (addop, 'max'  ) || ...
        isequal (addop, 'min' ) || isequal (addop, 'times'))
        return
    end
end

if (is_any_pair)
    fname = 'iso' ;
    unsigned = true ;
    bits = 0 ;
    zname = 'iso' ;
else
    [fname, unsigned, bits] = codegen_type (xytype) ;
    [zname, ~, ~] = codegen_type (ztype) ;
end

name = sprintf ('%s_%s_%s', addop, multop, fname) ;

% function names
fprintf (f, 'm4_define(`_Adot2B'', `_Adot2B__%s'')\n', name) ;
fprintf (f, 'm4_define(`_Adot3B'', `_Adot3B__%s'')\n', name) ;
fprintf (f, 'm4_define(`_Asaxpy3B'', `_Asaxpy3B__%s'')\n', name) ;
fprintf (f, 'm4_define(`_Asaxpy3B_M'', `_Asaxpy3B_M__%s'')\n', name) ;
fprintf (f, 'm4_define(`_Asaxpy3B_noM'', `_Asaxpy3B_noM__%s'')\n', name) ;
fprintf (f, 'm4_define(`_Asaxpy3B_notM'', `_Asaxpy3B_notM__%s'')\n', name) ;
fprintf (f, 'm4_define(`_AsaxbitB'', `_AsaxbitB__%s'')\n', name) ;
fprintf (f, 'm4_define(`GB_AxB'', `GB_AxB__%s'')\n', name) ;

% type of A, A2, B, B2, X, Y, Z, and C.  Pre-generated kernels have simple
% types: no typecasting, xtype and ytype are the same, and match the A and B
% types.  For the JIT, these types can differ.
if (is_any_pair)
    fprintf (f, 'm4_define(`GB_ztype'', `#define GB_Z_TYPE void'')\n') ;
    fprintf (f, 'm4_define(`GB_ctype'', `#define GB_C_TYPE void'')\n') ;
    fprintf (f, 'm4_define(`GB_c_iso'', `#define GB_C_ISO 1'')\n') ;
else
    fprintf (f, 'm4_define(`GB_ztype'', `#define GB_Z_TYPE %s'')\n', ztype) ;
    fprintf (f, 'm4_define(`GB_ctype'', `#define GB_C_TYPE %s'')\n', ztype) ;
    fprintf (f, 'm4_define(`GB_c_iso'', `#define GB_C_ISO 0'')\n') ;
end

% flag if ztype can ignore overflow in some computations
if (ztype_ignore_overflow)
    fprintf (f, 'm4_define(`GB_ztype_ignore_overflow'', `%s'')\n', ...
        '#define GB_Z_IGNORE_OVERFLOW 1') ;
else
    fprintf (f, 'm4_define(`GB_ztype_ignore_overflow'', `'')\n') ;
end

% identity value for the monoid
fprintf (f, 'm4_define(`GB_identity'', `%s'')\n', identity) ;
if (is_any_pair)
    define_id = '' ;
    define_const_id = '' ;
else
    define_id = sprintf (' %s z = %s', ztype, identity) ;
    define_const_id = sprintf (' const %s z = %s', ztype, identity) ;
end
fprintf (f, 'm4_define(`GB_declare_identity'', `#define GB_DECLARE_IDENTITY(z)%s'')\n', define_id) ;
fprintf (f, 'm4_define(`GB_declare_const_identity'', `#define GB_DECLARE_IDENTITY_CONST(z)%s'')\n', define_const_id) ;

if (is_any_pair)
    fprintf (f, 'm4_define(`GB_is_any_pair_semiring'', `%s'')\n', ...
        '#define GB_IS_ANY_PAIR_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_any_pair_semiring'', `'')\n') ;
end

if (is_plus_times_fp)
    % enable the avx-based methods.  only two semirings (plus_times_fp32 and
    % plus_times_fp64) are accelerated with AVX2 or AVX512f instructions.  More
    % semirings will be accelerated in the future.
    % This flag is also used to trigger the RISC-V RVV1.0 vectorization
    fprintf (f, 'm4_define(`if_semiring_has_avx'', `0'')\n') ;
    fprintf (f, 'm4_define(`GB_semiring_has_avx'', `#define GB_SEMIRING_HAS_AVX_IMPLEMENTATION 1'')\n') ;
else
    % disable the avx-based methods
    fprintf (f, 'm4_define(`if_semiring_has_avx'', `-1'')\n') ;
    fprintf (f, 'm4_define(`GB_semiring_has_avx'', `'')\n') ;
end

one = '' ;
if (is_pair)
    fprintf (f, 'm4_define(`GB_is_pair_multiplier'', `%s'')\n', ...
        '#define GB_IS_PAIR_MULTIPLIER 1') ;
    switch (ztype)
        case { 'GxB_FC32_t' }
            one = '#define GB_PAIR_ONE GxB_CMPLXF (1,0)' ;
        case { 'GxB_FC64_t' }
            one = '#define GB_PAIR_ONE GxB_CMPLX (1,0)' ;
        otherwise
    end
else
    fprintf (f, 'm4_define(`GB_is_pair_multiplier'', `'')\n') ;
end
fprintf (f, 'm4_define(`GB_pair_one'', `%s'')\n', one) ;

%-------------------------------------------------------------------------------
% very special cases for semirings
%-------------------------------------------------------------------------------

if (is_plus_pair_real)
    % plus_pair_(int or fp*), not complex 
    fprintf (f, 'm4_define(`GB_is_plus_pair_real_semiring'', `%s'')\n', ...
        '#define GB_IS_PLUS_PAIR_REAL_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_pair_real_semiring'', `'')\n') ;
end

if (is_lxor && is_pair)
    % xor_pair_bool
    fprintf (f, 'm4_define(`GB_is_lxor_pair_semiring'', `%s'')\n', ...
        '#define GB_IS_LXOR_PAIR_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_lxor_pair_semiring'', `'')\n') ;
end

if (is_plus && ztype_nbits == 8 && is_pair)
    % plus_pair_(int8, uint8)
    fprintf (f, 'm4_define(`GB_is_plus_pair_8_semiring'', `%s'')\n', ...
        '#define GB_IS_PLUS_PAIR_8_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_pair_8_semiring'', `'')\n') ;
end

if (is_plus && ztype_nbits == 16 && is_pair)
    % plus_pair_(int16, uint16)
    fprintf (f, 'm4_define(`GB_is_plus_pair_16_semiring'', `%s'')\n', ...
        '#define GB_IS_PLUS_PAIR_16_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_pair_16_semiring'', `'')\n') ;
end

if (is_plus && ztype_nbits == 32 && is_integer && is_pair)
    % plus_pair_(int32, uint32)
    fprintf (f, 'm4_define(`GB_is_plus_pair_32_semiring'', `%s'')\n', ...
        '#define GB_IS_PLUS_PAIR_32_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_pair_32_semiring'', `'')\n') ;
end

if (is_plus && is_pair && ...
    (isequal (ztype, 'int64_t') || ...
     isequal (ztype, 'uint64_t') || ...
     isequal (ztype, 'float') || ...
     isequal (ztype, 'double')))
    % plus_pair_(int64, uint64, float, double)
    fprintf (f, 'm4_define(`GB_is_plus_pair_big_semiring'', `%s'')\n', ...
        '#define GB_IS_PLUS_PAIR_BIG_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_pair_big_semiring'', `'')\n') ;
end

if (codegen_contains (ztype, 'GxB_FC'))
    fprintf (f, 'm4_define(`GB_ztype_is_complex'', `#define GB_Z_IS_COMPLEX 1'')\n') ;
else
    fprintf (f, 'm4_define(`GB_ztype_is_complex'', `'')\n') ;
end

%-------------------------------------------------------------------------------

if (is_any)
    fprintf (f, 'm4_define(`GB_is_any_monoid'', `%s'')\n', ...
        '#define GB_IS_ANY_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_any_monoid'', `'')\n') ;
end


% terminal value for the monoid
if (isempty (terminal))
    fprintf (f, 'm4_define(`GB_terminal'', `(no terminal value)'')\n', terminal) ;
else
    fprintf (f, 'm4_define(`GB_terminal'', `%s'')\n', terminal) ;
end

tcondition = '' ;
tbreak = '' ;
tvalue = '' ;

op = '' ;
if (is_any)
    % the ANY monoid terminates on the first entry seen
    is_terminal = 1 ;
elseif (~isempty (terminal))
    % terminal monoids terminate when cij equals the terminal value
    is_terminal = 1 ;
    tcondition = sprintf (' (z == %s)', terminal) ;
    tbreak = sprintf (' if (z == %s) { break ; }', terminal) ;
    tvalue = sprintf (' const %s zterminal = %s', ztype, terminal) ;
else
    % non-terminal monoids
    is_terminal = 0 ;
    if (ztype_is_real)
        switch (addop)
            case { 'plus' }
                op = '+' ;
            case { 'times' }
                op = '*' ;
            case { 'lor' }
                op = '||' ;
            case { 'land' }
                op = '&&' ;
            case { 'lxor' }
                op = '^' ;
            case { 'bor' }
                op = '|' ;
            case { 'band' }
                op = '&' ;
            case { 'bxor' }
                op = '^' ;
            otherwise
                op = '' ;
        end
    end
end

if (isempty (op))
    fprintf (f, 'm4_define(`GB_pragma_simd_reduction_monoid'', `'')\n') ;
else
    pragma = sprintf ('GB_PRAGMA_SIMD_REDUCTION (%s,cij)', op) ;
    fprintf (f, 'm4_define(`GB_pragma_simd_reduction_monoid'', `#define GB_PRAGMA_SIMD_REDUCTION_MONOID(cij) %s'')\n', pragma) ;
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
    % disable the ANY monoid for saxpy4
    fprintf (f, 'm4_define(`_Asaxpy4B'', `_Asaxpy4B__%s'')\n', '(none)') ;
    fprintf (f, 'm4_define(`if_saxpy4_enabled'', `-1'')\n') ;
else
    % enable saxpy4
    fprintf (f, 'm4_define(`_Asaxpy4B'', `_Asaxpy4B__%s'')\n', name) ;
    fprintf (f, 'm4_define(`if_saxpy4_enabled'', `0'')\n') ;
end

if (is_any)
    % dot4 is disabled for the ANY monoid
    fprintf (f, 'm4_define(`_Adot4B'', `_Adot4B__%s'')\n', '(none)') ;
    fprintf (f, 'm4_define(`if_dot4_enabled'', `-1'')\n') ;
else
    % enable dot4
    fprintf (f, 'm4_define(`_Adot4B'', `_Adot4B__%s'')\n', name) ;
    fprintf (f, 'm4_define(`if_dot4_enabled'', `0'')\n') ;
end

if (is_any)
    % saxpy5 is disabled for the ANY monoid
    fprintf (f, 'm4_define(`_Asaxpy5B'', `_Asaxpy5B__%s'')\n', '(none)') ;
    fprintf (f, 'm4_define(`if_saxpy5_enabled'', `-1'')\n') ;
else
    % enable saxpy5
    fprintf (f, 'm4_define(`_Asaxpy5B'', `_Asaxpy5B__%s'')\n', name) ;
    fprintf (f, 'm4_define(`if_saxpy5_enabled'', `0'')\n') ;
end

% firsti or firsti1 multiply operator
if (codegen_contains (multop, 'firsti'))
    fprintf (f, 'm4_define(`GB_is_firsti_multiplier'', `%s'')\n', ...
        '#define GB_IS_FIRSTI_MULTIPLIER 1 /* or FIRSTI1 */') ;
else
    fprintf (f, 'm4_define(`GB_is_firsti_multiplier'', `'')\n') ;
end

% firstj or firstj1 multiply operator
is_firstj = codegen_contains (multop, 'firstj') ;
if (is_firstj)
    fprintf (f, 'm4_define(`GB_is_firstj_multiplier'', `%s'')\n', ...
        '#define GB_IS_FIRSTJ_MULTIPLIER 1 /* or FIRSTJ1 */') ;
else
    fprintf (f, 'm4_define(`GB_is_firstj_multiplier'', `'')\n') ;
end

% secondj or secondj1 multiply operator
is_secondj = codegen_contains (multop, 'secondj') ;
if (is_secondj)
    fprintf (f, 'm4_define(`GB_is_secondj_multiplier'', `%s'')\n', ...
        '#define GB_IS_SECONDJ_MULTIPLIER 1 /* or SECONDJ1 */') ;
else
    fprintf (f, 'm4_define(`GB_is_secondj_multiplier'', `'')\n') ;
end

% offset for (first,second)*i1 or (first,second)*j1 multiply operator
if (codegen_contains (multop, 'i1') || codegen_contains (multop, 'j1'))
    fprintf (f, 'm4_define(`GB_offset'', `%s%s'')\n', ...
      '#define GB_OFFSET 1', ...
      ' /* offset for FIRSTI1, SECONDI1, FIRSTJ1, SECONDJ1 */') ;
else
    fprintf (f, 'm4_define(`GB_offset'', `'')\n') ;
end

% plus_fc32 monoid:
if (is_plus && isequal (ztype, 'GxB_FC32_t'))
    fprintf (f, 'm4_define(`GB_is_plus_fc32_monoid'', `%s'')\n', ...
        '#define GB_IS_PLUS_FC32_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_fc32_monoid'', `'')\n') ;
end

% plus_fc64 monoid:
if (is_plus && isequal (ztype, 'GxB_FC64_t'))
    fprintf (f, 'm4_define(`GB_is_plus_fc64_monoid'', `%s'')\n', ...
        '#define GB_IS_PLUS_FC64_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_plus_fc64_monoid'', `'')\n') ;
end

% min monoids:
is_imin = false ;
is_fmin = false ;
if (is_min)
    if (is_integer)
        % min monoid for signed or unsigned integers
        is_imin = true ;
    else
        % min monoid for float or double
        is_fmin = true ;
    end
else
    % not a min monoid
end

if (is_imin)
    fprintf (f, 'm4_define(`GB_is_imin_monoid'', `%s'')\n', ...
        '#define GB_IS_IMIN_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_imin_monoid'', `'')\n') ;
end

if (is_fmin)
    fprintf (f, 'm4_define(`GB_is_fmin_monoid'', `%s'')\n', ...
        '#define GB_IS_FMIN_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_fmin_monoid'', `'')\n') ;
end

if (is_imin && is_firstj)
    fprintf (f, 'm4_define(`GB_is_min_firstj_semiring'', `%s'')\n', ...
        '#define GB_IS_MIN_FIRSTJ_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_min_firstj_semiring'', `'')\n') ;
end

% max monoids:
is_imax = false ;
is_fmax = false ;
if (is_max)
    if (is_integer)
        % max monoid for signed or unsigned integers
        is_imax = true ;
    else
        % max monoid for float or double
        is_fmax = true ;
    end
else
    % not a max monoid
end

if (is_imax)
    fprintf (f, 'm4_define(`GB_is_imax_monoid'', `%s'')\n', ...
        '#define GB_IS_IMAX_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_imax_monoid'', `'')\n') ;
end

if (is_fmax)
    fprintf (f, 'm4_define(`GB_is_fmax_monoid'', `%s'')\n', ...
        '#define GB_IS_FMAX_MONOID 1') ;
else
    fprintf (f, 'm4_define(`GB_is_fmax_monoid'', `'')\n') ;
end

if (is_imax && is_firstj)
    fprintf (f, 'm4_define(`GB_is_max_firstj_semiring'', `%s'')\n', ...
        '#define GB_IS_MAX_FIRSTJ_SEMIRING 1') ;
else
    fprintf (f, 'm4_define(`GB_is_max_firstj_semiring'', `'')\n') ;
end

% to get an entry from A
if (is_second || is_pair || is_positional)
    % value of A is ignored for the SECOND, PAIR, and positional operators
    fprintf (f, 'm4_define(`GB_atype'',  `#define GB_A_TYPE void'')\n') ;
    fprintf (f, 'm4_define(`GB_a2type'', `#define GB_A2TYPE void'')\n') ;
    fprintf (f, 'm4_define(`GB_a_is_pattern'', `#define GB_A_IS_PATTERN 1'')\n') ;
    gb_geta = '' ;
    gb_declarea = '' ;
else
    fprintf (f, 'm4_define(`GB_atype'',  `#define GB_A_TYPE %s'')\n', xytype) ;
    fprintf (f, 'm4_define(`GB_a2type'', `#define GB_A2TYPE %s'')\n', xytype) ;
    fprintf (f, 'm4_define(`GB_a_is_pattern'', `'')\n') ;
    gb_geta = ' aik = Ax [(A_iso) ? 0 : (pA)]' ;
    gb_declarea = sprintf (' %s aik', xytype) ;
end
fprintf (f, 'm4_define(`GB_geta'', `#define GB_GETA(aik,Ax,pA,A_iso)%s'')\n', gb_geta) ;
fprintf (f, 'm4_define(`GB_declarea'', `#define GB_DECLAREA(aik)%s'')\n', gb_declarea) ;

% to get an entry from B
if (is_first || is_pair || is_positional)
    % value of B is ignored for the FIRST, PAIR, and positional operators
    fprintf (f, 'm4_define(`GB_btype'',  `#define GB_B_TYPE void'')\n') ;
    fprintf (f, 'm4_define(`GB_bsize'',  `#define GB_B_SIZE 0'')\n') ;
    fprintf (f, 'm4_define(`GB_b2type'', `#define GB_B2TYPE void'')\n') ;
    fprintf (f, 'm4_define(`GB_b_is_pattern'', `#define GB_B_IS_PATTERN 1'')\n') ;
    gb_getb = '' ;
    gb_declareb = '' ;
else
    fprintf (f, 'm4_define(`GB_btype'',  `#define GB_B_TYPE %s'')\n', xytype) ;
    fprintf (f, 'm4_define(`GB_bsize'',  `'')\n') ;
    fprintf (f, 'm4_define(`GB_b2type'', `#define GB_B2TYPE %s'')\n', xytype) ;
    fprintf (f, 'm4_define(`GB_b_is_pattern'', `'')\n') ;
    gb_getb = ' bkj = Bx [(B_iso) ? 0 : (pB)]' ;
    gb_declareb = sprintf (' %s bkj', xytype) ;
end
fprintf (f, 'm4_define(`GB_getb'', `#define GB_GETB(bkj,Bx,pB,B_iso)%s'')\n', gb_getb) ;
fprintf (f, 'm4_define(`GB_declareb'', `#define GB_DECLAREB(bkj)%s'')\n', gb_declareb) ;

% access the values of C
if (is_any_pair)
    fprintf (f, 'm4_define(`GB_putc'', `#define GB_PUTC(cij,Cx,p)'')\n') ;
else
    fprintf (f, 'm4_define(`GB_putc'', `#define GB_PUTC(cij,Cx,p) Cx [p] = cij'')\n') ;
end

% type-specific idiv
if (~isempty (strfind (mult, 'idiv')))
    if (unsigned)
        mult = strrep (mult, 'idiv', sprintf ('idiv_uint%d', bits)) ;
    else
        mult = strrep (mult, 'idiv', sprintf ('idiv_int%d', bits)) ;
    end
end

% create the multiply operator (assignment)
if (is_any_pair)
    mult2_defn = sprintf ('#define GB_MULT(z,a,b,i,k,j)    ') ;
else
    mult2 = strrep (mult,  'xarg', 'a') ;
    mult2 = strrep (mult2, 'yarg', 'b') ;
    mult2_defn = sprintf ('#define GB_MULT(z,a,b,i,k,j)    z = %s', mult2) ;
end
fprintf (f, 'm4_define(`GB_multiply'', `%s'')\n', mult2_defn) ;

% create the add_update, of the form z += t
if (is_any_pair)
    gb_update = '' ;
elseif (is_min)
    if (is_integer)
        % min monoid for signed or unsigned integers
        gb_update = 'if (z > t) { z = t ; }' ;
    else
        % min monoid for float or double, with omitnan property
        if (t_is_nonnan)
            gb_update = 'if (!islessequal (z, t)) { z = t ; }' ;
        else
            gb_update = 'if (!isnan (t) && !islessequal (z, t)) { z = t ; }' ;
        end
    end
elseif (is_max)
    if (is_integer)
        % max monoid for signed or unsigned integers
        gb_update = 'if (z < t) { z = t ; }' ;
    else
        % max monoid for float or double, with omitnan property
        if (t_is_nonnan)
            gb_update = 'if (!isgreaterequal (z, t)) { z = t ; }' ;
        else
            gb_update = 'if (!isnan (t) && !isgreaterequal (z, t)) { z = t ; }';
        end
    end
else
    % use the update function as given but convert warg and targ
    gb_update = strrep (update,    'warg', 'z') ;
    gb_update = strrep (gb_update, 'targ', 't') ;
end
gb_update = sprintf ('#define GB_UPDATE(z,t)          %s', gb_update) ;
fprintf (f, 'm4_define(`GB_add_update'', `%s'')\n', gb_update) ;

% create the GB_ADD function for the monoid
if (is_any_pair)
    gb_add_op = '' ;
else
    gb_add_op = strrep (addfunc,   'zarg', 'z') ;
    gb_add_op = strrep (gb_add_op, 'xarg', 'zin') ;
    gb_add_op = strrep (gb_add_op, 'yarg', 't') ;
end
gb_add_op = sprintf ('#define GB_ADD(z,zin,t)         %s', gb_add_op) ;
fprintf (f, 'm4_define(`GB_add_op'', `%s'')\n', gb_add_op) ;

% create the fused multiply-add statement, of the form: z += x*y ;
is_imin_or_imax = (is_min || is_max) && is_integer ;
if (is_any_pair)
    multadd = '' ;
    % fprintf (f, 'm4_define(`GB_multiply_add'', `'')\n') ;
elseif (~is_imin_or_imax && ...
    (isequal (ztype, 'float') || isequal (ztype, 'double') || ...
     isequal (ztype, 'bool') || is_first || is_second || is_pair || is_positional))
    % float and double do not get promoted.
    % bool is OK since promotion of the result (0 or 1) to int is safe.
    % first and second are OK since no promotion occurs.
    % positional operators are OK too.
    multadd = strrep (update, 'targ',  mult) ;
    multadd = strrep (multadd, 'warg', 'z') ;
    multadd = strrep (multadd, 'xarg', 'a') ;
    multadd = strrep (multadd, 'yarg', 'b') ;
    % fprintf (f, 'm4_define(`GB_multiply_add'', `%s'')\n', multadd) ;
else
    % use explicit typecasting to avoid ANSI C integer promotion.
    update2 = strrep (update,  'warg', 'z') ;
    update2 = strrep (update2, 'targ', 'x_op_y') ;
    multadd = sprintf ('{ %s x_op_y = %s ; %s ; }', ztype, mult2, update2) ;
end
multadd = sprintf ('#define GB_MULTADD(z,a,b,i,k,j) %s', multadd) ;
fprintf (f, 'm4_define(`GB_multiply_add'', `%s'')\n', multadd) ;

% determine the identity byte
idbyte = '' ;

switch (addop)

    % any monoid
    case { 'any' }

    % boolean monoids (except eq / lxnor)
    case { 'lor' }
        idbyte = '0' ;
    case { 'land' }
        idbyte = '1' ;
    case { 'lxor' }
        idbyte = '0' ;

    % min/max monoids:
    case { 'min' }
        if (codegen_contains (ztype, 'uint'))
            idbyte = '0xFF' ;
        elseif (isequal (ztype, 'int8_t'))
            idbyte = '0x7F' ;
        end
    case { 'max' }
        if (codegen_contains (ztype, 'uint'))
            idbyte = '0' ;
        elseif (isequal (ztype, 'int8_t'))
            idbyte = '0x80' ;
        end

    % plus monoid:
    case { 'plus' }
        idbyte = '0' ;

    % bitwise monoids (except bxnor)
    case { 'bor' }
        idbyte = '0' ;
    case { 'band' }
        idbyte = '0xFF' ;
    case { 'bxor' }
        idbyte = '0' ;

    case { 'eq' }
        if (ztype_nbits == 8)
            idbyte = '1' ;
        end
    case { 'times' }
        if (ztype_nbits == 8)
            idbyte = '1' ;
        end
    case {'bxnor' }
        idbyte = '0xFF' ;
end

if (isempty (idbyte))
    fprintf (f, 'm4_define(`GB_has_identity_byte'', `'')\n') ;
    fprintf (f, 'm4_define(`GB_identity_byte'', `'')\n') ;
else
    fprintf (f, 'm4_define(`GB_has_identity_byte'', `%s'')\n', ...
        '#define GB_HAS_IDENTITY_BYTE 1') ;
    sbyte = sprintf ('#define GB_IDENTITY_BYTE %s', idbyte) ;
    fprintf (f, 'm4_define(`GB_identity_byte'', `%s'')\n', sbyte) ;
end

% create the disable flag
if (is_any_pair)
    % never disable the any_pair_iso semiring
    fprintf (f, 'm4_define(`GB_disable'', `#define GB_DISABLE 0'')\n') ;
    fprintf (f, 'm4_define(`GB_type_enabled'', `#define GB_TYPE_ENABLED 1'')\n');
else
    codegen_type_enabled (f, fname) ;
    disable  = sprintf ('defined(GxB_NO_%s)', upper (addop)) ;
    if (~isequal (addop, multop))
        disable = [disable (sprintf (' || defined(GxB_NO_%s)', upper (multop)))] ;
    end
    disable = [disable (sprintf (' || defined(GxB_NO_%s)', upper (fname)))] ;
    disable = [disable (sprintf (' || defined(GxB_NO_%s_%s)', upper (addop), upper (zname)))] ;
    if (~ (isequal (addop, multop) && isequal (zname, fname)))
        disable = [disable (sprintf (' || defined(GxB_NO_%s_%s)', upper (multop), upper (fname)))] ;
    end
    disable = [disable (sprintf (' || defined(GxB_NO_%s_%s_%s)', ...
        upper (addop), upper (multop), upper (fname))) ] ;
    fprintf (f, 'm4_define(`GB_disable'', `#if (%s)\n#define GB_DISABLE 1\n#else\n#define GB_DISABLE 0\n#endif\n'')\n', disable) ;
end

fprintf (f, 'm4_divert(0)\n') ;
fclose (f) ;

if (is_any_pair)
    % the ANY_PAIR_ISO semiring goes in Source/mxm
    s = '../mxm' ;
    k = 1 ;
else
    % all other semirings go in ../../FactoryKernels
    s = '../../FactoryKernels' ;
    k = 2 ;
end

% construct the *.c file for the semiring
cmd = sprintf ('cat control.m4 Generator/GB_AxB.c | m4 -P | awk -f codegen_blank.awk > %s/GB_AxB__%s.c', s, name) ;
system (cmd) ;

fprintf ('.') ;

% append to the *.h file
cmd = sprintf ('cat control.m4 Generator/GB_AxB.h | m4 -P | awk -f codegen_blank.awk | grep -v SPDX >> %s/GB_AxB__include%d.h', s, k) ;
system (cmd) ;

delete ('control.m4') ;

