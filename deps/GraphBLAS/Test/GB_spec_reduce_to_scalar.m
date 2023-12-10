function c = GB_spec_reduce_to_scalar (cin, accum, reduce, A)
%GB_SPEC_REDUCE_TO_SCALAR a mimic of GrB_reduce (to scalar)
%
% Usage:
% c = GB_spec_reduce_to_scalar (cin, accum, reduce, A)
%
% Reduces a matrix or vector to a scalar
%
% cin is a dense scalar

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargout > 1 || nargin ~= 4)
    error ('usage: c = GB_spec_Matrix_reduce (cin, accum, reduce, A)') ;
end

if (isstruct (cin) || issparse (cin) || ~isscalar (cin))
    error ('cin must be a dense scalar') ;
end

cin_type = GB_spec_type (cin) ;

% get the type of A
if (isstruct (A))
    atype = A.class ;
else
    atype = GB_spec_type (A) ;
end

% get the reduce operator. default type is the type of A
[reduce_op reduce_optype zt xt yt] = GB_spec_operator (reduce, atype) ;
assert (isequal (zt, xt)) ;
assert (isequal (zt, yt)) ;

if (GB_spec_is_positional (reduce_op))
    error ('reduce operator cannot be positional') ;
end

% get the identity
identity = GB_spec_identity (reduce_op, reduce_optype) ;
if (isempty (identity))
    identity = 0 ;
end

% get the input matrix
A = GB_spec_matrix (A, identity) ;

% get the accumulator and its types for z = accum(x,y)
[accum_op accum_optype ztype xtype ytype ] = GB_spec_operator (accum, cin_type) ;

if (GB_spec_is_positional (accum_op))
    error ('accum operator cannot be positional') ;
end

%-------------------------------------------------------------------------------
% do the work via a clean *.m interpretation of the entire GraphBLAS spec
%-------------------------------------------------------------------------------

% cast A to the type of the reduce operator, but only entries in the pattern

t = identity ;
if (nnz (A.pattern) > 0)
    X = GB_mex_cast (A.matrix (A.pattern), reduce_optype) ;
    for k = 1:length (X)
        t = GB_spec_op (reduce, t, X (k)) ;
    end
end

% apply the accumulator.  no mask. t is a dense matrix so this is
% different than GB_spec_accum.
if (isempty (accum_op))
    c = GB_mex_cast (t, cin_type) ;
else
    % c = cin "+" t, but typecast all the arguments
    t   = GB_mex_cast (t,   xtype) ;     % cast t to accum xtype
    cin = GB_mex_cast (cin, ytype) ;     % cast cin to accum ytype
    z = GB_spec_op (accum, cin, t) ;     % z = accum (cin,t)
    c = GB_mex_cast (z, cin_type) ;      % cast z to type of cin
end

