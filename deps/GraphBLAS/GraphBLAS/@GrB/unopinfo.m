function unopinfo (op, type)
%GRB.UNOPINFO list the details of a GraphBLAS unary operator.
%
%   GrB.unopinfo
%   GrB.unopinfo (op)
%   GrB.unopinfo (op, type)
%
% For GrB.unopinfo(op), the op must be a string of the form 'op.type',
% where 'op' is listed below.  The second usage allows the type to be
% omitted from the first argument, as just 'op'.  This is valid for all
% GraphBLAS operations, since the type defaults to the type of the input
% matrix.  However, GrB.unopinfo does not have a default type and thus one
% must be provided, either in the op as GrB.unopinfo ('abs.double'), or in
% the second argument, GrB.unopinfo ('abs', 'double').
%
% The functions z=f(x) are listed below.  Unless otherwise specified,
% z and x have the same type.  Some functions have synonyms, as listed.
%
% For all 13 types:
%   identity    z = x       also '+', 'uplus'
%   ainv        z = -x      additive inverse, also '-', 'negate', 'uminus'
%   minv        z = 1/x     multiplicative inverse
%   one         z = 1       does not depend on x, also '1'
%   abs         z = |x|     'abs.complex' returns a real result
%
% For all 11 real types:
%   lnot        z = ~(x ~= 0)   logical negation (z is 1 or 0, with the
%                               same type as x), also '~', 'not'.
%
% For 4 floating-point types (real & complex)x(single & double):
%   sqrt        z = sqrt (x)    square root
%   log         z = log (x)     base-e logarithm
%   log2        z = log2 (x)    base-2 logarithm
%   log10       z = log10 (x)   base-10 logarithm
%   log1p       z = log1p (x)   log (x-1), base-e
%   exp         z = exp (x)     base-e exponential, e^x
%   pow2        z = pow2 (x)    base-2 exponential, 2^x
%   expm1       z = exp1m (x)   e^x-1
%   sin         z = sin (x)     sine
%   cos         z = cos (x)     cosine
%   tan         z = tan (x)     tangent
%   acos        z = acos (x)    arc cosine
%   asin        z = asin (x)    arc sine
%   atan        z = atan (x)    arc tangent
%   sinh        z = sinh (x)    hyperbolic sine
%   cosh        z = cosh (x)    hyperbolic cosine
%   tanh        z = tanh (x)    hyperbolic tangent
%   asinh       z = asinh (x)   inverse hyperbolic sine
%   acosh       z = acosh (x)   inverse hyperbolic cosine
%   atanh       z = atanh (x)   inverse hyperbolic tangent
%   signum      z = signum (x)  signum function, also 'sign'
%   ceil        z = ceil (x)    ceiling
%   floor       z = floor (x)   floor
%   round       z = round (x)   round to nearest
%   trunc       z = trunc (x)   truncate, also 'fix'
%
% For 'single complex' and 'double complex' only:
%   creal       z = real (x)    real part of x (z is real), also 'real'
%   cimag       z = imag (x)    imag. part of x (z is real), also 'imag'
%   carg        z = carg (x)    phase angle (z is real), also 'angle'
%   conj        z = conj (x)    complex conjugate (z is complex)
%
% For all 4 floating-point types (result is logical):
%   isinf       z = isinf (x)       true if x is +Inf or -Inf
%   isnan       z = isnan (x)       true if x is NaN
%   isfinite    z = isfinite (x)    true if x is finite
%
% For single and double (result same type as input):
%   lgamma      z = lgamma (x)  log of gamma function, also 'gammaln'
%   tgamma      z = tgamma (x)  gamma function, also 'gamma'
%   erf         z = erf (x)     error function
%   erfc        z = erfc (x)    complementary error function
%   cbrt        z = cbrt (x)    cube root
%   frexpx      z = frexpx (x)  mantissa from C11 frexp function
%   frexpe      z = frexpe (x)  exponent from C11 frexp function;
%                               the built-in [f,e]=log2(x) returns
%                               f = frexpx (x) and e = frexpe (x).
%
% For integer types only (result is same type as input):
%   bitcmp      z = ~(x)        bitwise complement, also 'bitnot'
%
% For int32 and int64 types, applied to an entry A(i,j)
%   positioni0  z = i-1     also 'i0'
%   positioni1  z = i       also 'i', 'i1', and 'positioni'
%   positionj0  z = j-1     also 'j0'
%   positionj1  z = j       also 'j', 'j1', and 'positionj'
%
% Example:
%
%   % valid unary operators
%   GrB.unopinfo ('+.double') ;     % also a valid binary operator
%   GrB.unopinfo ('abs.double') ;
%   GrB.unopinfo ('not.int32') ;
%   GrB.unopinfo ('pow2.double') ;  % also a valid binary operator
%   GrB.binopinfo ('pow2.double') ;
%
%   % invalid unary operator (generates an error; this is a binary op):
%   GrB.unopinfo ('*.double') ;
%
% See also GrB.binopinfo, GrB.descriptorinfo, GrB.monoidinfo,
% GrB.selectopinfo, GrB.semiringinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    help GrB.unopinfo
elseif (nargin == 1)
    gbunopinfo (op) ;
else
    gbunopinfo (op, type) ;
end

