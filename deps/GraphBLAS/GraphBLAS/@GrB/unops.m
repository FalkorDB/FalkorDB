function result = unops
%GRB.BINOPS list all unary ops
%
% Example:
%   GrB.unops ;             % prints a list, with descriptions
%   list = GrB.unops ;      % returns the list (nothing printed)
%
% See also GrB.unopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

types = {
    'logical'
    'double'
    'single'
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    'single complex'
    'double complex'
    } ;

ops = {
    'identity'  , 'f(x) = x' ;
    '~'         , 'f(x) = ~x, where z has the same type as x' ;
    '-'         , 'f(x) = -x' ;
    '1'         , 'f(x) = 1' ;
    'minv'      , 'f(x) = 1/x' ;
    'abs'       , 'absolute value' ;
    'sqrt'      , 'square root' ;
    'log'       , 'base-e logarithm' ;
    'exp'       , 'base-e exponential, e^x' ;
    'sin'       , 'sine' ;
    'cos'       , 'cosine' ;
    'tan'       , 'tangent' ;
    'asin'      , 'arc sine' ;
    'acos'      , 'arc cosine' ;
    'atan'      , 'arc tangent' ;
    'sinh'      , 'hyperbolic sine' ;
    'cosh'      , 'hyperbolic cosine' ;
    'tanh'      , 'hyperbolic tangent' ;
    'asinh'     , 'hyperbolic arc sine' ;
    'acosh'     , 'hyperbolic arc cosine' ;
    'atanh'     , 'hyperbolic arc tangent' ;
    'sign'      , 'signum function' ;
    'ceil'      , 'round to +inf' ;
    'floor'     , 'round to -inf' ;
    'round'     , 'round to nearest integer' ;
    'fix'       , 'round to zero' ;
    'pow2'      , '2^x' ;
    'expm1'     , '(e^x)-1' ;
    'log10'     , 'base-10 logarithm' ;
    'log1p'     , 'log(1+x)' ;
    'log2'      , 'base-2 logarithm' ;
    'gammaln'   , 'log of gamma' ;
    'gamma'     , 'gamma function' ;
    'erf'       , 'error function' ;
    'cbrt'      , 'cube root' ;
    'erfc'      , 'complementary error function' ;
    'conj'      , 'complex conjugate' ;
    'creal'     , 'real part of a complex number' ;
    'cimag'     , 'imaginary part of a complex number' ;
    'angle'     , 'complex phase angle' ;
    'isinf'     , 'true if +inf or -inf' ;
    'isnan'     , 'true if nan' ;
    'isfinite'  , 'true if finite (not inf, -inf, or nan)' ;
    'frexpx'    , 'normalized fractional part, in range [.5,1)' ;
    'frexpe'    , 'integral exponent; x = frexpx(x)*2^frexpe(x)' ;
    'bitcmp'    , 'bitwise complement' ;
    'i0'        , 'row index of the entry in its matrix, minus 1' ;
    'i1'        , 'row index of the entry in its matrix' ;
    'j0'        , 'column index of the entry in its matrix, minus 1' ;
    'j1'        , 'column index of the entry in its matrix' ;
    } ;

nunops = 0 ;
nops = size (ops, 1) ;

if (nargout > 0)
    result = { } ;
end

for k1 = 1:nops
    op = ops {k1,1} ;
    op_description = ops {k1,2} ;
    first_op = true ;
    for k2 = 1:length (types)
        ok = false ;
        type = types {k2}  ;
        unop = [op '.' type] ;
        try
            ok = gbunopinfo (unop) ;
            nunops = nunops + 1 ;
            if (nargout > 0)
                result = [result ; unop] ; %#ok<AGROW>
            end
        catch
        end

        if (ok && nargout == 0)
            if (first_op)
                fprintf ('unary op: %s.type', op) ;
                fprintf (', %s\n', op_description) ;
                fprintf ('        types: %s', type) ;
            else
                fprintf (', %s', type) ;
            end
            first_op = false ;
        end
    end
    if (nargout == 0)
        fprintf ('\n\n') ;
    end

end

if (nargout == 0)
    fprintf ('Total number of available unary ops: %d\n', nunops) ;
end


