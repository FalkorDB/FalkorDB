% @GrB interface for SuiteSparse:GraphBLAS
%
% GraphBLAS is a library for creating graph algorithms based on sparse
% linear algebraic operations over semirings.  Its @GrB interface
% provides faster sparse matrix operations than the built-in methods,
% as well as sparse integer and single-precision matrices, and
% operations with arbitrary semirings.  See 'help GrB' for details.
%
% The constructor method is GrB.  If A is any matrix (GraphBLAS,
% or built-in sparse or full), then:
%
%   C = GrB (A) ;            GraphBLAS copy of a matrix A, same type
%   C = GrB (m, n) ;         m-by-n GraphBLAS double matrix with no entries
%   C = GrB (..., type) ;    create or typecast to a different type
%   C = GrB (..., format) ;  create in a specified format
%
% The type can be 'double', 'single', 'logical', 'int8', 'int16', 'int32',
% 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'double complex' or
% 'single complex'.  Typical formats are 'by row' or 'by col'. 
%
% Essentially all operators and many built-in functions are
% overloaded by the @GrB class, so that they can be used for GraphBLAS
% matrices.  See 'help GrB' for more details.
%
% Tim Davis, Texas A&M University,
% http://faculty.cse.tamu.edu/davis/GraphBLAS
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

