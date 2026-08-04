function C = gb_cat (ghb, dim, varargin)
%GB_CAT implements GrB/cat and GhB/cat.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% concatenate the matrices
if (dim == 1)
    % same as vertcat
    C = gzb_cat (ghb, varargin') ;
else
    % same as horzcat
    C = gzb_cat (ghb, varargin) ;
end


