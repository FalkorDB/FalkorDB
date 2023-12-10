function [p, varargout] = symamd (G, varargin)
%SYMAMD approximate minimum degree ordering.
% See 'help symamd' for details.
%
% See also GrB/amd, GrB/colamd, GrB/symrcm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[p, varargout{1:nargout-1}] = symamd (double (G), varargin {:}) ;

