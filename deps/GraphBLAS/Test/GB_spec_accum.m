function [Z simple] = GB_spec_accum (accum, C, T, identity)
%GB_SPEC_ACCUM mimic of the Z=accum(C,T) operation in GraphBLAS
%
% Z = GB_spec_accum (accum, C, T, identity)
%
% Apply accum binary operator to the input C and the intermediate result T.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% get the operator; of is type(C) if type is not present in the accum op
[opname optype ztype xtype ytype] = GB_spec_operator (accum, C.class) ;

if (GB_spec_is_positional (opname))
    error ('accum operator cannot be positional') ;
end

if (nargin < 4)
    identity = 0 ;
end

% simple is true if the simpler accum_mask.m script will work without
% any typecast, and with a zero identity
simple = (identity == 0) ;

% initialize the matrix Z, same size and class as C
[nrows ncols] = size (C.matrix) ;
Z.matrix  = GB_spec_zeros ([nrows ncols], C.class) ;
% Z.matrix (:,:) = GB_mex_cast (identity, C.class) ;
Z.pattern = false (nrows, ncols) ;
Z.class = C.class ;

if (isempty (opname))

    % Z = T, casting into the class of C
    p = T.pattern ;
    Z.matrix (p) = GB_mex_cast (T.matrix (p), C.class) ;
    Z.pattern = T.pattern ;

    simple = simple && isfield (T, 'class') && isequal (T.class, C.class) ;

else

    % Z = accum (C,T)

    % apply the operator to entries in the intersection of C and T
    p = T.pattern & C.pattern ;
    % first cast the entries into the class of the operator
    c = GB_mex_cast (C.matrix (p), xtype) ;
    t = GB_mex_cast (T.matrix (p), ytype) ;
    z = GB_spec_op (accum, c, t) ;
    % cast the result z from optype into the class of C
    Z.matrix (p) = GB_mex_cast (z, C.class) ;

    % copy entries in C but not in T, into the result Z, no typecasting needed
    p = C.pattern & ~T.pattern ;
    Z.matrix (p) = C.matrix (p) ;

    % cast entries in T but not in C, into the result Z
    p = T.pattern & ~C.pattern ;
    Z.matrix (p) = GB_mex_cast (T.matrix (p), C.class) ;

    % the pattern of Z is the union of both T and C
    Z.pattern = C.pattern | T.pattern ;

    simple = simple && ( ...
        isequal (C.class, optype) && ...
        isfield (T, 'class') && ...
        isequal (T.class, optype) && ...
        isequal (GB_spec_type (z), C.class) && ...
        isequal (T.class, C.class)) ;
end

