function assert (G)
%ASSERT generate an error when a condition is violated.
%
% See also error.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

builtin ('assert', logical (G)) ;

