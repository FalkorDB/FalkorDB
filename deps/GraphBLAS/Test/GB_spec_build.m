function [S,p] = GB_spec_build (I, J, X, nrows, ncols, op, order, sclass)
%GB_SPEC_BUILD a version of GrB_Matrix_build and GrB_vector_build
%
% Usage:
% [S p] = GB_spec_build (I, J, X, nrows, ncols, op, order)
%
% GB_spec_build builds a full matrix S, mimicing GB_mex_Matrix_build but using
% almost pure built-in methods.  This function is very slow since it creates a
% dense matrix instead of a sparse one.  It is meant only as an executable
% version of the GraphBLAS spec.  It cannot operate purely with built-in
% methods, however, because the casting and operator rules for built-in methods
% differ from the C-style casting and operator rules GraphBLAS.  With built-in
% methods, adding two int8 values 120 + 30 results in 127; since 150 is larger
% than the max int8 value of 127, the result is that max value.  In C, the
% result wraps around, modulo 256, to be -106.

%
% S is returned as a struct, with S.matrix being the matrix, S.class the class,
% and S.pattern the nonzero pattern of S.
%
% I: row indices. Indices are zero-based, just like GB_mex_Matrix_build.
%
% optional arguments:
% J: column indices. Default J is a vector of all zeros, for building a column
%       vector.
% X: numerical values, with built-in class logical, any integer, single, or
%       double.  I, J, and X must have the same number of entries.
%       Default X is a logical vector of all true.
% nrows: number of rows of S.  Default is nrows = max (I) + 1 ;
% ncols: number of cols of S.  Default is ncols = max (J) + 1 ;
% op: binary operator z=f(x,y) for assembling duplicates.  See
%       GB_spec_operator.  The GraphBLAS spec requires op to be associative
%       (min, max, plus, or times) but any binary operator with x,y,z
%       types the same will work; see the 'order' parameter.
% order: 'natural', or 'random'.  Default is 'natural'.
%       The GraphBLAS spec does not state what order the duplicates are
%       assembled.  It only guarantees the result if op is associative.  The
%       'natural' order sums them up in the order they appear in [I,J,X], and
%       this is what GB_mex_Matrix_build does.  To check the results against
%       any possible order, use 'random', which sums them up in a random
%       order.  The results of 'natural' and 'random' will differ if op is not
%       associative.  With floating-point values, roundoff may vary slightly,
%       which should be acceptable.  If the results differ greatly then the
%       problem is very ill-conditioned.  The output p returns the ordering
%       used to assemble the entries, a permutation of 1:length(X).
%
% Use an empty value ([ ] or '') to obtain the default value for optional
% parameters, or pass fewer inputs.  For exampe S = GB_spec_build (I, J, X,
% nrows, ncols) uses defaults for op, and order, but not X, nrows and ncols.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 2 || nargin < 1 || nargin > 8)
    error ('usage: [S p] = GB_spec_build (I, J, X, nrows, ncols, op, order, sclass)') ;
end

% get I
nnz = length (I) ;

% get J
if (nargin < 2)
    J = [ ] ;
end
if (isempty (J))
    J = zeros (nnz, 1) ;
end
if (nnz ~= length (J))
    error ('I and J must have the same size') ;
end

% get X
if (nargin < 3)
    X = [ ] ;
end
if (isempty (X))
    X = ones (nnz, 1, 'logical') ;
end
if (nnz ~= length (X))
    error ('I and X must have the same size') ;
end

% get the number of rows
if (nargin < 4)
    nrows = [ ] ;
end
if (isempty (nrows))
    nrows = max (I) + 1 ;
end

% get the number of cols
if (nargin < 5)
    ncols = [ ]
end
if (isempty (ncols))
    ncols = max (J) + 1 ;
end

% get the op
if (nargin < 6)
    op = [ ] ;
end
if (isempty (op))
    op = 'plus' ;
end
[opname optype ztype xtype ytype] = GB_spec_operator (op, GB_spec_type (X)) ;

if (GB_spec_is_positional (opname))
    error ('dup operator cannot be positional') ;
end

assert (isequal (ztype, xtype)) ;
assert (isequal (ztype, ytype)) ;

% get the ordering
if (nargin < 7)
    order = '' ;
end

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

% sort or randomize the tuples
if (isequal (order, 'random'))
    % the GraphBLAS spec allows for any ordering
    p = randperm (nnz) ;
else
    % the 'natural' ordering is used in this implementation of GrB_Matrix_build
    [~, p] = sortrows ([J I (1:nnz)']) ;
end
I = I (p) ;
J = J (p) ;
X = X (p) ;

% initialize the matrix S and its pattern
S.matrix = GB_spec_zeros ([nrows ncols], optype) ;
S.pattern = false (nrows, ncols) ;
S.class = optype ;

% assemble the tuples into S
for t = 1:nnz
    i = 1 + I (t) ;     % convert from 0-based GraphBLAS to 1-based
    j = 1 + J (t) ;
    if (~S.pattern (i,j))
        % first time S(i,j) is modified: cast x into S
        S.matrix (i,j) = GB_mex_cast (X (t), optype) ;
        S.pattern (i,j) = true ;
    else
        % a duplicate entry to be assembled with the operator op
        % cast x into the class of S and the operator
        x = GB_mex_cast (X (t), optype) ;
        % apply the operator, result is of class optype
        S.matrix (i,j) = GB_spec_op (op, S.matrix (i,j), x) ;
    end
end

% get the sclass
if (nargin < 8)
    sclass = optype ;  % default is optype
end

% typecast S into the desired class
if (~isequal (optype, sclass))
    S.matrix = GB_mex_cast (S.matrix, sclass) ;
    S.class = sclass ;
end


