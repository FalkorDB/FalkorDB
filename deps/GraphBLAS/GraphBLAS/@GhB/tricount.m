function s = tricount (A, arg2, arg3)
%GHB.TRICOUNT count triangles in a matrix.
% s = GhB.tricount (A) is the number of triangles in the matrix A.  spones (A)
% must be symmetric; results are undefined if spones (A) is unsymmetric.
% Diagonal entries are ignored.
%
% To check the input matrix A, use GhB.tricount (A, 'check').  This check takes
% additional time so by default the input is not checked.
%
% If d is a vector of length n with d(i) equal to the degree of node i, then s
% = tricount (A, d) can be used.  Otherwise, tricount must compute the degrees
% first.
%
% See also GhB.ktruss, GhB.entries.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

d = [ ] ;
check = false ;

if (nargin == 2)
    if (ischar (arg2))
        % s = tricount (A, 'check')
        check = isequal (arg2, 'check') ;
    else
        % s = tricount (A, d)
        d = arg2 ;
    end
elseif (nargin == 3)
    if (ischar (arg2))
        % s = tricount (A, 'check', d)
        check = isequal (arg2, 'check') ;
        d = arg3 ;
    else
        % s = tricount (A, d, 'check')
        d = arg2 ;
        check = isequal (arg3, 'check') ;
    end
end

if (check && ~issymmetric (spones (A)))
    error ('GrB:error', 'pattern of A must be symmetric') ;
end

if (isobject (d))
    d = double (d) ;
end

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

% determine if A should be sorted first
nsamples = 2000 ;
if (n > nsamples && GhB.entries (A) >= 10*n)
    if (isempty (d))
        % compute the degree of each node, if not provided on input
        if (GhB.isbyrow (A))
            d = double (GhB.entries (A, 'row', 'degree')) ;
        else
            d = double (GhB.entries (A, 'col', 'degree')) ;
        end
    end
    % sample the degree
    sample = d (randperm (n, nsamples)) ;
    dmean = full (mean (sample)) ;
    dmed  = full (median (sample)) ;
    if (dmean > 3 * dmed)
        % sort if the average degree is very high compared to the median
        [~, p] = sort (d, 'ascend') ;
        % S = logical (A (p,p))
        p = { p } ;
        S = GhB (n, n, 'logical') ;
        GhB.extract (S, A, p, p) ;
        clear p
    else
        % use A as-is
        S = A ;
    end
else
    % use A as-is
    S = A ;
end

% determine the type for C and the semiring
type = 'int64' ;
if (n < 2^31)
    type = 'int32' ;
end
semiring = ['+.oneb.' type] ;

%-------------------------------------------------------------------------------
% construct L and U
%-------------------------------------------------------------------------------

% C, L, and U have the same format as S
C = GhB (n, n, type, GhB.format (S)) ;
L = tril (S, -1) ;
U = triu (S, 1) ;

% Inside GraphBLAS, the methods below are identical.  For example, L stored by
% row is the same data structure as U stored by column.  Both use the
% Sandia_LUT method as defined in LAGraph (case 5), which is typically the
% fastest of the methods in LAGraph_tricount.

desc.mask = 'structural' ;

if (GhB.isbyrow (S))
    % C<L> = L*U'
    desc.in1 = 'transpose' ;
    GhB.mxm (C, L, semiring, L, U, desc) ;
else
    % C<U> = L'*U; same as Sandia_LUT when all matrices are held by column
    desc.in0 = 'transpose' ;
    GhB.mxm (C, U, semiring, L, U, desc) ;
end

s = full (GhB.reduce ('+.int64', C)) ;

