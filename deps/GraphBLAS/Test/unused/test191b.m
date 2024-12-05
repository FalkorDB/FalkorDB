function test191b(tasks)
%TEST191B test split

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test191 ----------- Tiles = split (A)\n') ;

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {1,        'logical', 1, 0}, ... % (  1,   1)
    {1,        'logical', 4, 0}, ... % (  1,   2)
    {2,        'logical', 8, 0}, ... % (  1,   3)
    } ;
end

test191 (tasks) ;

