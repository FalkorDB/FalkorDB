function gbtest116 (ghb)
%GBTEST116 list all idxunop operators for [GrB,GhB].apply2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

opnames = {
    'tril',
    'triu',
    'diag',
    'offdiag',
    'diagindex',
    'rowindex',
    'rowle',
    'rowgt',
    'colindex',
    'colle',
    'colgt' } ;

for k1 = 1:length(opnames)
    op = opnames {k1} ;
    fprintf ('\n=================================== %s\n', op) ;
    gtb_binopinfo (ghb, op) ;
end

fprintf ('gbtest116 (%d): all tests passed\n', ghb) ;

