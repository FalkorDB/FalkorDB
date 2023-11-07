function gbtest71
%GBTEST71 test GrB.selectopinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

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
    GrB.selectopinfo (ops {k}) ;
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
            GrB.selectopinfo (ops {k1}, types {k2}) ;
        end
    end
end

fprintf ('\n\n') ;
GrB.selectopinfo

fprintf ('gbtest71: all tests passed\n') ;

