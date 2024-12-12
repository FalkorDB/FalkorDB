function test238b(tasks)
%TEST238B test GrB_mxm (dot4 and dot2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    { 1, 1, 1, 2}, ... % (  8,   8)
    { 1, 1, 1, 3}, ... % (  3,  11)
    { 1, 1, 1, 4}, ... % (  1,  12)
    } ;
end

test238 (tasks) ;

fprintf ('\ntest238: all tests passed\n') ;


