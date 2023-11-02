function [complex_binaryops complex_unaryops ] = GB_user_opsall
%GB_USER_OPSALL return list of complex operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% complex binary operators
complex_binaryops = {
% 11 operators where x,y,z are all the same class
'first',     % z = x
'second',    % z = y
'pair',      % z = 1
'oneb',      % z = 1 (same as pair)
'plus',      % z = x + y
'minus',     % z = x - y
'rminus',    % z = y - x
'times',     % z = x * y
'div',       % z = x / y
'rdiv',      % z = y / x
% comparators where x,y,z are all the same class
'iseq',      % z = (x == y)
'isne',      % z = (x != y)
%----------------------------
% comparators where x,y are all the same class, z is logical
'eq',        % z = (x == y)
'ne',        % z = (x != y)
%----------------------------
'complex'    % z = complex (x,y)
} ;

complex_unaryops = {
% 6 where x,z are complex
'one',         % z = 1
'identity',    % z = x
'ainv',        % z = -x
'abs',         % z = abs(x) (z is real)
'minv'         % z = 1/x
'conj'         % z = conj(x)
%----------------------------
% 4 where x is complex, z is double
'real'         % z = real(x)
'imag'         % z = imag(x)
'carg'         % z = angle(x)
} ;

