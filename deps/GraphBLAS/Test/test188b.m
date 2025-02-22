function test188b(tasks)
%TEST188B test concat

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: the test coverage requires the JIT cache to be empty first,
% for full coverage.

fprintf ('test188b ----------- C = concat (Tiles)\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {       'logical',         'double', 1, 0, 1, 0, 0}, ... % (1, 1)
    {       'logical',         'double', 2, 0, 1, 0, 0}, ... % (0, 1)
    {       'logical',         'double', 3, 0, 1, 0, 0}, ... % (1, 2)
    {          'int8',        'logical', 4, 0, 8, 0, 0}} ;   % (1, 3)
end

test188 (tasks) ;

