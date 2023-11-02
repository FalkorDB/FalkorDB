function codegen_unop
%CODEGEN_UNOP create functions for all unary operators
%
% This function creates all files of the form GB_unop__*.[ch],
% and the include file GB_unop__include.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\nunary operators:\n') ;

fh = fopen ('FactoryKernels/GB_unop__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_unop__include.h: definitions for GB_unop__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_unop.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

codegen_unop_identity ;

codegen_unop_template ('ainv', ...
    'xarg',                     ... % bool
    '-xarg',                    ... % int
    '-xarg',                    ... % uint
    '-xarg',                    ... % float
    '-xarg',                    ... % double
    'GB_FC32_ainv (xarg)',      ... % GxB_FC32_t
    'GB_FC64_ainv (xarg)') ;    ... % GxB_FC64_t

codegen_unop_template ('abs', ...
    'xarg',                     ... % bool
    'GB_IABS (xarg)',           ... % int
    'xarg',                     ... % uint
    'fabsf (xarg)',             ... % float
    'fabs (xarg)',              ... % double
    [ ],                        ... % GxB_FC32_t (see below)
    [ ]) ;                      ... % GxB_FC64_t (see below)

codegen_unop_template ('minv', ...
    'true',                     ... % bool
    'GB_iminv (xarg)',          ... % int
    'GB_iminv (xarg)',          ... % uint
    '(1.0F)/xarg',              ... % float
    '1./xarg',                  ... % double
    'GB_FC32_div (GxB_CMPLXF (1,0), xarg)', ... % GxB_FC32_t
    'GB_FC64_div (GxB_CMPLX  (1,0), xarg)') ;   % GxB_FC64_t

codegen_unop_template ('lnot',  ...
    '!xarg',                    ... % bool
    '!(xarg != 0)',             ... % int
    '!(xarg != 0)',             ... % uint
    '!(xarg != 0)',             ... % float
    '!(xarg != 0)',             ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('bnot',  ...
    [ ],                        ... % bool
    '~(xarg)',                  ... % int
    '~(xarg)',                  ... % uint
    [ ],                        ... % float
    [ ],                        ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('sqrt', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'sqrtf (xarg)',             ... % float
    'sqrt (xarg)',              ... % double
    'GB_csqrtf (xarg)',            ... % GxB_FC32_t
    'GB_csqrt (xarg)') ;           ... % GxB_FC64_t

codegen_unop_template ('log', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'logf (xarg)',              ... % float
    'log (xarg)',               ... % double
    'GB_clogf (xarg)',             ... % GxB_FC32_t
    'GB_clog (xarg)') ;            ... % GxB_FC64_t

codegen_unop_template ('exp', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'expf (xarg)',              ... % float
    'exp (xarg)',               ... % double
    'GB_cexpf (xarg)',             ... % GxB_FC32_t
    'GB_cexp (xarg)') ;            ... % GxB_FC64_t

codegen_unop_template ('sin', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'sinf (xarg)',              ... % float
    'sin (xarg)',               ... % double
    'GB_csinf (xarg)',             ... % GxB_FC32_t
    'GB_csin (xarg)') ;            ... % GxB_FC64_t

codegen_unop_template ('cos', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'cosf (xarg)',              ... % float
    'cos (xarg)',               ... % double
    'GB_ccosf (xarg)',             ... % GxB_FC32_t
    'GB_ccos (xarg)') ;            ... % GxB_FC64_t

codegen_unop_template ('tan', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'tanf (xarg)',              ... % float
    'tan (xarg)',               ... % double
    'GB_ctanf (xarg)',             ... % GxB_FC32_t
    'GB_ctan (xarg)') ;            ... % GxB_FC64_t

codegen_unop_template ('asin', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'asinf (xarg)',             ... % float
    'asin (xarg)',              ... % double
    'GB_casinf (xarg)',            ... % GxB_FC32_t
    'GB_casin (xarg)') ;           ... % GxB_FC64_t

codegen_unop_template ('acos', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'acosf (xarg)',             ... % float
    'acos (xarg)',              ... % double
    'GB_cacosf (xarg)',            ... % GxB_FC32_t
    'GB_cacos (xarg)') ;           ... % GxB_FC64_t

codegen_unop_template ('atan', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'atanf (xarg)',             ... % float
    'atan (xarg)',              ... % double
    'GB_catanf (xarg)',            ... % GxB_FC32_t
    'GB_catan (xarg)') ;           ... % GxB_FC64_t


codegen_unop_template ('sinh', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'sinhf (xarg)',             ... % float
    'sinh (xarg)',              ... % double
    'GB_csinhf (xarg)',            ... % GxB_FC32_t
    'GB_csinh (xarg)') ;           ... % GxB_FC64_t

codegen_unop_template ('cosh', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'coshf (xarg)',             ... % float
    'cosh (xarg)',              ... % double
    'GB_ccoshf (xarg)',            ... % GxB_FC32_t
    'GB_ccosh (xarg)') ;           ... % GxB_FC64_t

codegen_unop_template ('tanh', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'tanhf (xarg)',             ... % float
    'tanh (xarg)',              ... % double
    'GB_ctanhf (xarg)',            ... % GxB_FC32_t
    'GB_ctanh (xarg)') ;           ... % GxB_FC64_t

codegen_unop_template ('asinh', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'asinhf (xarg)',            ... % float
    'asinh (xarg)',             ... % double
    'GB_casinhf (xarg)',           ... % GxB_FC32_t
    'GB_casinh (xarg)') ;          ... % GxB_FC64_t

codegen_unop_template ('acosh', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'acoshf (xarg)',            ... % float
    'acosh (xarg)',             ... % double
    'GB_cacoshf (xarg)',           ... % GxB_FC32_t
    'GB_cacosh (xarg)') ;          ... % GxB_FC64_t

codegen_unop_template ('atanh', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'atanhf (xarg)',            ... % float
    'atanh (xarg)',             ... % double
    'GB_catanhf (xarg)',           ... % GxB_FC32_t
    'GB_catanh (xarg)') ;          ... % GxB_FC64_t

codegen_unop_template ('signum', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'GB_signumf (xarg)',        ... % float
    'GB_signum (xarg)',         ... % double
    'GB_csignumf (xarg)',       ... % GxB_FC32_t
    'GB_csignum (xarg)') ;      ... % GxB_FC64_t

codegen_unop_template ('ceil', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'ceilf (xarg)',             ... % float
    'ceil (xarg)',              ... % double
    'GB_cceilf (xarg)',         ... % GxB_FC32_t
    'GB_cceil (xarg)') ;        ... % GxB_FC64_t

codegen_unop_template ('floor', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'floorf (xarg)',            ... % float
    'floor (xarg)',             ... % double
    'GB_cfloorf (xarg)',        ... % GxB_FC32_t
    'GB_cfloor (xarg)') ;       ... % GxB_FC64_t

codegen_unop_template ('round', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'roundf (xarg)',            ... % float
    'round (xarg)',             ... % double
    'GB_croundf (xarg)',        ... % GxB_FC32_t
    'GB_cround (xarg)') ;       ... % GxB_FC64_t

codegen_unop_template ('trunc', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'truncf (xarg)',            ... % float
    'trunc (xarg)',             ... % double
    'GB_ctruncf (xarg)',        ... % GxB_FC32_t
    'GB_ctrunc (xarg)') ;       ... % GxB_FC64_t

codegen_unop_template ('exp2', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'exp2f (xarg)',             ... % float
    'exp2 (xarg)',              ... % double
    'GB_cexp2f (xarg)',         ... % GxB_FC32_t
    'GB_cexp2 (xarg)') ;        ... % GxB_FC64_t

codegen_unop_template ('expm1', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'expm1f (xarg)',            ... % float
    'expm1 (xarg)',             ... % double
    'GB_cexpm1f (xarg)',        ... % GxB_FC32_t
    'GB_cexpm1 (xarg)') ;       ... % GxB_FC64_t

codegen_unop_template ('log10', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'log10f (xarg)',            ... % float
    'log10 (xarg)',             ... % double
    'GB_clog10f (xarg)',        ... % GxB_FC32_t
    'GB_clog10 (xarg)') ;       ... % GxB_FC64_t

codegen_unop_template ('log1p', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'log1pf (xarg)',            ... % float
    'log1p (xarg)',             ... % double
    'GB_clog1pf (xarg)',        ... % GxB_FC32_t
    'GB_clog1p (xarg)') ;       ... % GxB_FC64_t

codegen_unop_template ('log2', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'log2f (xarg)',             ... % float
    'log2 (xarg)',              ... % double
    'GB_clog2f (xarg)',         ... % GxB_FC32_t
    'GB_clog2 (xarg)') ;        ... % GxB_FC64_t

codegen_unop_template ('frexpx', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'GB_frexpxf (xarg)',        ... % float
    'GB_frexpx (xarg)',         ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('frexpe', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'GB_frexpef (xarg)',        ... % float
    'GB_frexpe (xarg)',         ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('lgamma', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'lgammaf (xarg)',           ... % float
    'lgamma (xarg)',            ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('tgamma', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'tgammaf (xarg)',           ... % float
    'tgamma (xarg)',            ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('erf', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'erff (xarg)',              ... % float
    'erf (xarg)',               ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('erfc', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'erfcf (xarg)',             ... % float
    'erfc (xarg)',              ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('cbrt', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    'cbrtf (xarg)',             ... % float
    'cbrt (xarg)',              ... % double
    [ ],                        ... % GxB_FC32_t
    [ ]) ;                      ... % GxB_FC64_t

codegen_unop_template ('conj', ...
    [ ],                        ... % bool
    [ ],                        ... % int
    [ ],                        ... % uint
    [ ],                        ... % float
    [ ],                        ... % double
    'GB_conjf (xarg)',             ... % GxB_FC32_t
    'GB_conj (xarg)') ;            ... % GxB_FC64_t

%-------------------------------------------------------------------------------
% z = f(x) where the type of z and x differ
%-------------------------------------------------------------------------------

% z = abs (x): x is complex, z is real
fprintf ('\nabs      ') ;
codegen_unop_method ('abs', 'GB_cabsf (xarg)', 'float' , 'GxB_FC32_t') ;
codegen_unop_method ('abs', 'GB_cabs (xarg)' , 'double', 'GxB_FC64_t') ;

% z = creal (x): x is complex, z is real
fprintf ('\ncreal    ') ;
codegen_unop_method ('creal', 'GB_crealf (xarg)', 'float' , 'GxB_FC32_t') ;
codegen_unop_method ('creal', 'GB_creal (xarg)' , 'double', 'GxB_FC64_t') ;

% z = cimag (x): x is complex, z is real
fprintf ('\ncimag    ') ;
codegen_unop_method ('cimag', 'GB_cimagf (xarg)', 'float' , 'GxB_FC32_t') ;
codegen_unop_method ('cimag', 'GB_cimag (xarg)' , 'double', 'GxB_FC64_t') ;

% z = carg (x): x is complex, z is real
fprintf ('\ncarg     ') ;
codegen_unop_method ('carg', 'GB_cargf (xarg)', 'float' , 'GxB_FC32_t') ;
codegen_unop_method ('carg', 'GB_carg (xarg)' , 'double', 'GxB_FC64_t') ;

% z = isinf (x): x is floating-point, z is bool
fprintf ('\nisinf    ') ;
codegen_unop_method ('isinf', 'isinf (xarg)'     , 'bool', 'float') ;
codegen_unop_method ('isinf', 'isinf (xarg)'     , 'bool', 'double') ;
codegen_unop_method ('isinf', 'GB_cisinff (xarg)', 'bool', 'GxB_FC32_t') ;
codegen_unop_method ('isinf', 'GB_cisinf (xarg)' , 'bool', 'GxB_FC64_t') ;

% z = isnan (x): x is floating-point, z is bool
fprintf ('\nisnan    ') ;
codegen_unop_method ('isnan', 'isnan (xarg)'     , 'bool', 'float') ;
codegen_unop_method ('isnan', 'isnan (xarg)'     , 'bool', 'double') ;
codegen_unop_method ('isnan', 'GB_cisnanf (xarg)', 'bool', 'GxB_FC32_t') ;
codegen_unop_method ('isnan', 'GB_cisnan (xarg)' , 'bool', 'GxB_FC64_t') ;

% z = isfinite (x): x is floating-point, z is bool
fprintf ('\nisfinite ') ;
codegen_unop_method ('isfinite', 'isfinite (xarg)'     , 'bool', 'float') ;
codegen_unop_method ('isfinite', 'isfinite (xarg)'     , 'bool', 'double') ;
codegen_unop_method ('isfinite', 'GB_cisfinitef (xarg)', 'bool', 'GxB_FC32_t') ;
codegen_unop_method ('isfinite', 'GB_cisfinite (xarg)' , 'bool', 'GxB_FC64_t') ;
fprintf ('\n') ;

