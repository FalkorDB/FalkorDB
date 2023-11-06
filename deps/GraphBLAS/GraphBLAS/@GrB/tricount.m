function s = tricount (A, arg2, arg3)
%GRB.TRICOUNT count triangles in a matrix.
% s = GrB.tricount (A) is the number of triangles in the matrix A.
% spones (A) must be symmetric; results are undefined if spones (A) is
% unsymmetric.  Diagonal entries are ignored.
%
% To check the input matrix A, use GrB.tricount (A, 'check').  This check
% takes additional time so by default the input is not checked.
%
% If d is a vector of length n with d(i) equal to the degree of node i,
% then s = tricount (A, d) can be used.  Otherwise, tricount must compute
% the degrees first.
%
% See also GrB.ktruss, GrB.entries.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% NOTE: this is a high-level algorithm that uses GrB objects.

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

if (isequal (class (d), 'GrB'))
    d = double (d) ;
end

% determine if A should be sorted first
if (n > 1000 && GrB.entries (A) >= 10*n)
    if (isempty (d))
        % compute the degree of each node, if not provided on input
        if (GrB.isbyrow (A))
            d = double (GrB.entries (A, 'row', 'degree')) ;
        else
            d = double (GrB.entries (A, 'col', 'degree')) ;
        end
    end
    % sample the degree
    sample = d (randperm (n, 1000)) ;
    dmean = full (mean (sample)) ;
    dmed  = full (median (sample)) ;
    if (dmean > 4 * dmed)
        % sort if the average degree is very high compared to the median
        [~, p] = sort (d, 'descend') ;
        % A = A (p,p) ;
        A = GrB.extract (A, { p }, { p }) ;
        clear p
    end
end

% C, L, and U will have the same format as A
C = GrB (n, n, 'int64', GrB.format (A)) ;
L = tril (A, -1) ;
U = triu (A, 1) ;

% Inside GraphBLAS, the methods below are identical.  For example, L stored by
% row is the same data structure as U stored by column.  Both use the
% SandiaDot2 method as defined in LAGraph (case 6), which is typically the
% fastest of the methods in LAGraph_tricount.

desc.mask = 'structural' ;

if (GrB.isbyrow (A))
    % C<U> = U*L': SandiaDot2 method
    desc.in1 = 'transpose' ;
    C = GrB.mxm (C, U, '+.oneb.int64', U, L, desc) ;
else
    % C<U> = L'*U: SandiaDot2 method
    desc.in0 = 'transpose' ;
    C = GrB.mxm (C, U, '+.oneb.int64', L, U, desc) ;
end

s = full (double (GrB.reduce ('+.int64', C))) ;

