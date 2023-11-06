function p = amd (G, varargin)
%AMD approximate minimum degree ordering.
% See 'help amd' for details.
%
% See also GrB/colamd, GrB/symrcm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

p = builtin ('amd', logical (G), varargin {:}) ;

