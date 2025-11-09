function s = sprintf (varargin)
%SPRINTF write formatted data to a string.
% The GraphBLAS sprintf function is identical to the built-in function;
% this overloaded method simply typecasts any GraphBLAS matrices to
% built-in matrices first, and then calls the builtin sprintf.
%
% See also fprintf, sprintf, GrB/fprintf.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = gb_printf_helper ('sprintf', varargin {:}) ;

