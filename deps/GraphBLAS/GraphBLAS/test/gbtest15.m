function gbtest15 (ghb)
%GBTEST15 list all unary operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

types = gbtest_types ;
ops = { 'identity', '~', '-', '1', 'minv', 'abs',  'sqrt', 'log', ...
    'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', ...
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', ...
    'signum', 'ceil', 'floor', 'round', 'trunc', 'pow2', ...
    'expm1', 'log10', 'log1p', 'log2', 'lgamma', 'tgamma', 'erf', ...
    'cbrt', ...
    'erfc', 'conj', 'creal', 'cimag', 'carg', 'isinf', 'isnan', ...
    'isfinite', 'frexpx', 'frexpe', 'bitnot', 'i0', 'i1', 'j0', 'j1' } ;

nops = 0 ;
for k1 = 1:length (ops)
    for k2 = 1:length (types)
        op = [ops{k1} '.' types{k2}] ;
        fprintf ('\nop: (%s)\n', op) ;
        try
            gtb_unopinfo (ghb, op) ;
            gtb_unopinfo (ghb, ops {k1}, types {k2}) ;
            nops = nops + 1 ;
        catch
        end
    end
end

fprintf ('\nhelp %s.unopinfo:\n', gtb_name) ;
gtb_unopinfo (ghb) ;

fprintf ('number of unary ops: %d\n', nops) ;
assert (nops == 226) ;

fprintf ('gbtest15 (%d): all tests passed\n', ghb) ;

