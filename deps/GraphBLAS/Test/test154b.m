function test154b(tasks)
%TEST154B test GrB_apply with scalar binding

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {     'plus',          'int16',  1, 1,-1, 1}, ... % (  1,    1)
    {     'plus',          'int16',  4, 4,-1, 1}, ... % (  1,    2)
    } ;
end

test154 (tasks) ;


