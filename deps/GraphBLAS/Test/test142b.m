function test142b(tasks)
%TEST142B test GrB_assign for dense matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    tasks = [ ] ;
end

if (isempty (tasks))
    tasks = {
    {          '',        'logical',  1}, ... % (  5,   5)
    {    'second',        'logical',  1}, ... % (  1,   6)
    {          '', 'single complex',  1}, ... % (  1,   7)
    {      'pair', 'single complex',  2}, ... % (  1,   8)
    {      'pair', 'double complex',  2}, ... % (  1,   9)
    } ;
end

test142 (tasks) ;

