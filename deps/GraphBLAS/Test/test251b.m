function test251b(tasks)
%TEST251B test dot4 for plus-pair semirings
% GB_AxB_dot4 computes C+=A'*B when C is dense.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: test coverage should start with an empty JIT cache.

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    { 0, 1, 1}, ... % ( 32,  32)
    { 1, 1, 1}, ... % (  4,  36)
    { 2, 1, 1}, ... % (  9,  45)
    { 4, 1, 1}, ... % (  4,  49)
    { 5, 1, 1}, ... % (  8,  57)
    { 6, 1, 1}, ... % (  4,  61)
    { 8, 1, 1}, ... % (  4,  65)
    { 9, 1, 1}, ... % (  4,  69)
    { 7, 2, 1}, ... % (  2,  71)
    } ;
end

test251 (tasks) ;

fprintf ('\n') ;
fprintf ('test251b: all tests passed\n') ;


