function gbtest71 (ghb)
%GBTEST71 test [GrB,GhB].selectopinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

ops = {
    'tril'
    'triu'
    'diag'
    'offdiag'
    'rowne'
    'rowle'
    'rowgt'
    'colne'
    'colle'
    'colgt'
    '~=0'
    'nonzero'
    '==0'
    'zero'
    '>0'
    'positive'
    '>=0'
    'nonnegative'
    '<0'
    'negative'
    '<=0'
    'nonpositive'
    '~='
    '=='
    '>'
    '>='
    '<'
    '<=' } ;

nops = length (ops) ;
for k = 1:nops
    gtb_selectopinfo (ghb, ops {k}) ;
end

ops = {
    '~='
    '=='
    '>'
    '>='
    '<'
    '<=' } ;
nops = length (ops) ;

types = gbtest_types ;
ntypes = length (types) ;

for k1 = 1:nops
    fprintf ('\n-------------- %s with specific types:\n', ops {k1}) ;
    for k2 = 1:ntypes
        if (gb_contains (types {k2}, 'complex') && k1 > 2)
            % skip this
        else
            gtb_selectopinfo (ghb, ops {k1}, types {k2}) ;
        end
    end
end

fprintf ('\n\n') ;
gtb_selectopinfo (ghb)

fprintf ('gbtest71 (%d): all tests passed\n', ghb) ;

