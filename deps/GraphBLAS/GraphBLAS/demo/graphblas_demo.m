%% GraphBLAS: graph algorithms in the language of linear algebra
% GraphBLAS is a library for creating graph algorithms based on sparse
% linear algebraic operations over semirings.  Visit http://graphblas.org
% for more details and resources.  See also the SuiteSparse:GraphBLAS
% User Guide in this package.
%
% http://faculty.cse.tamu.edu/davis
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%% GraphBLAS: faster and more general sparse matrices for MATLAB
% GraphBLAS is not only useful for creating graph algorithms; it also
% supports a wide range of sparse matrix data types and operations.
% MATLAB can compute C=A*B with just two semirings: 'plus.times.double'
% and 'plus.times.complex' for complex matrices.  GraphBLAS has 2,518
% built-in semirings, such as 'max.plus'
% (https://en.wikipedia.org/wiki/Tropical_semiring).  These semirings can
% be used to construct a wide variety of graph algorithms, based on
% operations on sparse adjacency matrices.
%
% MATLAB and GraphBLAS both provide sparse matrices of type double,
% logical, and double complex.  GraphBLAS adds sparse matrices of type:
% single, int8, int16, int32, int64, uint8, uint16, uint32, uint64, and
% single complex (with MATLAB matrices, these types can only be held in
% full matrices).

% reset to the default number of threads
clear all
ncores = demo_nproc ;
GrB.clear ;
fprintf ('\n# of threads used by GraphBLAS: %d\n', GrB.threads) ;

format compact
rng ('default') ;
X = 100 * rand (2) ;
G = GrB (X)              % GraphBLAS copy of a matrix X, same type

%% Sparse integer matrices
% Here's an int8 version of the same matrix:

S = int8 (G)             % convert G to a full built-in int8 matrix
S (1,1) = 0              % add an explicit zero to S
G = GrB (X, 'int8')      % a GraphBLAS full int8 matrix
G (1,1) = 0              % add an explicit zero to G
G = GrB.prune (G)        % a GraphBLAS sparse int8 matrix

try
    S = sparse (S) ;     % built-in sparse matrices cannot be int8
catch me
    display (me)
end

%% Sparse single-precision matrices
% Matrix operations in GraphBLAS are typically as fast, or faster than
% MATLAB.  The following test is unfair, since it compares computing
% X*X with MATLAB in double precision and with GraphBLAS in single
% precision.  You would naturally expect GraphBLAS to be faster. 
%
% CAVEAT:  MATLAB R2021a uses SuiteSparse:GraphBLAS v3.3.3 for C=A*B,
% so on that version of MATLAB, we're comparing 2 versions of GraphBLAS
% by the same author.
%
% Please wait ...

n = 1e5 ;
X = spdiags (rand (n, 201), -100:100, n, n) ;
G = GrB (X, 'single') ;
tic
G2 = G*G ;
gb_time = toc ;
tic
X2 = X*X ;
builtin_time = toc ;
fprintf ('\nGraphBLAS time: %g sec (in single)\n', gb_time) ;
fprintf ('%s time:    %g sec (in double)\n', demo_whoami, builtin_time) ;
fprintf ('Speedup of GraphBLAS over %s: %g\n', ...
    demo_whoami, builtin_time / gb_time) ;
fprintf ('\n# of threads used by GraphBLAS: %d\n', GrB.threads) ;

%% Mixing MATLAB and GraphBLAS matrices
% The error in the last computation is about eps('single') since
% GraphBLAS did its computation in single precision, while MATLAB used
% double precision.  MATLAB and GraphBLAS matrices can be easily
% combined, as in X2-G2.  The sparse single precision matrices take less
% memory space.

err = norm (X2 - G2, 1) / norm (X2,1)
eps ('single')
whos G G2 X X2

%% Faster matrix operations
% But even with standard double precision sparse matrices, GraphBLAS is
% typically faster than the built-in MATLAB methods.  Here's a fair
% way to compare (caveat: these both use GraphBLAS in MATLAB R2021a):

G = GrB (X) ;
tic
G2 = G*G ;
gb_time = toc ;
err = norm (X2 - G2, 1) / norm (X2,1)
fprintf ('\nGraphBLAS time: %g sec (in double)\n', gb_time) ;
fprintf ('%s time:    %g sec (in double)\n', demo_whoami, builtin_time) ;
fprintf ('Speedup of GraphBLAS over %s: %g\n', ...
    demo_whoami, builtin_time / gb_time) ;
fprintf ('\n# of threads used by GraphBLAS: %d\n', GrB.threads) ;

%% A wide range of semirings
% MATLAB can only compute C=A*B using the standard '+.*.double' and
% '+.*.complex' semirings.  A semiring is defined in terms of a string,
% 'add.mult.type', where 'add' is a monoid that takes the place of the
% additive operator, 'mult' is the multiplicative operator, and 'type' is
% the data type for the two inputs to the mult operator.
%
% In the standard semiring, C=A*B is defined as:
%
%   C(i,j) = sum (A(i,:).' .* B(:,j))
%
% using 'plus' as the monoid and 'times' as the multiplicative operator.
% But in a more general semiring, 'sum' can be any monoid, which is an
% associative and commutative operator that has an identity value.  For
% example, in the 'max.plus' tropical algebra, C(i,j) for C=A*B is
% defined as:
%
%   C(i,j) = max (A(i,:).' + B(:,j))

%%
% This can be computed in GraphBLAS with:
%
%   C = GrB.mxm ('max.+', A, B)

n = 3 ;
A = rand (n) ;
B = rand (n) ;
C = zeros (n) ;
for i = 1:n
    for j = 1:n
        C(i,j) = max (A (i,:).' + B (:,j)) ;
    end
end
C2 = GrB.mxm ('max.+', A, B) ;
fprintf ('\nerr = norm (C-C2,1) = %g\n', norm (C-C2,1)) ;

%% The max.plus tropical semiring
% Here are details of the "max.plus" tropical semiring.  The identity
% value is -inf since max(x,-inf) = max (-inf,x) = x for any x.
% The identity for the conventional "plus.times" semiring is zero,
% since x+0 = 0+x = x for any x.

GrB.semiringinfo ('max.+.double') ;

%% A boolean semiring
% MATLAB cannot multiply two logical matrices.  MATLAB R2019a converts
% them to double and uses the conventional +.*.double semiring instead.
% In GraphBLAS, this is the common Boolean 'or.and.logical' semiring,
% which is widely used in linear algebraic graph algorithms.

GrB.semiringinfo ('|.&.logical') ;

%%
clear
A = sparse (rand (3) > 0.5)
B = sparse (rand (3) > 0.2)

%%
try
    % MATLAB R2019a does this by casting A and B to double
    C1 = A*B
catch
    % MATLAB R2018a throws an error
    fprintf ('MATLAB R2019a required for C=A*B with logical\n') ;
    fprintf ('matrices.  Explicitly converting to double:\n') ;
    C1 = double (A) * double (B)
end
C2 = GrB (A) * GrB (B)

%%
% Note that C1 is a MATLAB sparse double matrix, and contains non-binary
% values.  C2 is a GraphBLAS logical matrix.
whos
GrB.type (C2)

%% GraphBLAS operators, monoids, and semirings
% The C interface for SuiteSparse:GraphBLAS allows for arbitrary types
% and operators to be constructed.  However, the MATLAB interface to
% SuiteSparse:GraphBLAS is restricted to pre-defined types and operators:
% a mere 13 types, 212 unary operators, 401 binary operators, 77 monoids,
% 22 select operators (each of which can be used for all 13 types),
% and 2,518 semirings.
%
% That gives you a lot of tools to create all kinds of interesting
% graph algorithms.  For example:
%
%   GrB.dnn    % sparse deep neural network (http://graphchallenge.org)
%   GrB.mis    % maximal independent set
%
% See 'help GrB.binopinfo' for a list of the binary operators, and
% 'help GrB.monoidinfo' for the ones that can be used as the additive
% monoid in a semiring.  'help GrB.unopinfo' lists the unary operators.
% 'help GrB.semiringinfo' describes the semirings.

%% 
help GrB.binopinfo

%% 
help GrB.monoidinfo

%% 
help GrB.unopinfo

%% 
help GrB.semiringinfo

%% Element-wise operations
% Binary operators can be used in element-wise matrix operations, like
% C=A+B and C=A.*B.  For the matrix addition C=A+B, the pattern of C is
% the set union of A and B, and the '+' operator is applied for entries
% in the intersection.  Entries in A but not B, or in B but not A, are
% assigned to C without using the operator.  The '+' operator is used for
% C=A+B but any operator can be used with GrB.eadd.

%%
A = GrB (sprand (3, 3, 0.5)) ;
B = GrB (sprand (3, 3, 0.5)) ;
C1 = A + B
C2 = GrB.eadd ('+', A, B)
err = norm (C1-C2,1)

%% Subtracting two matrices
% A-B and GrB.eadd ('-', A, B) are not the same thing, since the '-'
% operator is not applied to an entry that is in B but not A.

C1 = A-B 
C2 = GrB.eadd ('-', A, B)

%% 
% But these give the same result.  GrB.eunion applies the operator
% as op(alpha,B) when A(i,j) is not present but B(i,j) is, and
% as op(A,beta) when A(i,j) is present but B(i,j) is not.
% In this case, both alpha and beta are zero.

C1 = A-B 
C2 = GrB.eadd ('+', A, GrB.apply ('-', B))
C3 = GrB.eunion ('-', A, 0, B, 0)
err = norm (C1-C2,1)
err = norm (C1-C3,1)

%% Element-wise 'multiplication'
% For C = A.*B, the result C is the set intersection of the pattern of A
% and B.  The operator is applied to entries in both A and B.  Entries in
% A but not B, or B but not A, do not appear in the result C.

C1 = A.*B
C2 = GrB.emult ('*', A, B) 
C3 = double (A) .* double (B)

%%
% Just as in GrB.eadd and GrB.eunion, any operator can be used in GrB.emult:

A
B
C2 = GrB.emult ('max', A, B) 

%% Overloaded operators
% The following operators all work as you would expect for any matrix.
% The matrices A and B can be GraphBLAS matrices, or MATLAB sparse or
% dense matrices, in any combination, or scalars where appropriate,
% The matrix M is logical (MATLAB or GraphBLAS):
%
%    A+B   A-B  A*B   A.*B  A./B  A.\B  A.^b   A/b   C=A(I,J)  C(M)=A
%    -A    +A   ~A    A'    A.'   A&B   A|B    b\A   C(I,J)=A  C=A(M)
%    A~=B  A>B  A==B  A<=B  A>=B  A<B   [A,B]  [A;B] C(A)
%    A(1:end,1:end)
%
% For A^b, b must be a non-negative integer.

C1 = [A B] ;
C2 = [double(A) double(B)] ;
assert (isequal (double (C1), C2))

%%
C1 = A^2
C2 = double (A)^2 ;
err = norm (C1 - C2, 1)
assert (err < 1e-12)

%%
C1 = A (1:2,2:end)
A = double (A) ;
C2 = A (1:2,2:end) ;
assert (isequal (double (C1), C2))

%% Overloaded functions
% Many MATLAB built-in functions can be used with GraphBLAS matrices:
%
% A few differences with the built-in functions:
%
%   S = sparse (G)        % converts G to sparse/hypersparse
%   F = full (G)          % adds explicit zeros, so numel(F)==nnz(F)
%   F = full (G,type,id)  % adds explicit identity values to a GrB matrix
%   disp (G, level)       % display a GrB matrix G; level=2 is the default.
%
% In the list below, the first set of Methods are overloaded built-in
% methods.  They are used as-is on GraphBLAS matrices, such as C=abs(G).
% The Static methods are prefixed with "GrB.", as in C = GrB.apply ( ... ).

%%

methods GrB

%% Zeros are handled differently
% Explicit zeros cannot be automatically dropped from a GraphBLAS matrix,
% like they are in MATLAB sparse matrices.  In a shortest-path problem,
% for example, an edge A(i,j) that is missing has an infinite weight,
% (the monoid identity of min(x,y) is +inf).  A zero edge weight A(i,j)=0
% is very different from an entry that is not present in A.  However, if
% a GraphBLAS matrix is converted into a MATLAB sparse matrix, explicit
% zeros are dropped, which is the convention for a MATLAB sparse matrix.
% They can also be dropped from a GraphBLAS matrix using the GrB.select
% method.

%%

G = GrB (magic (2)) ;
G (1,1) = 0      % G(1,1) still appears as an explicit entry
A = double (G)   % but it's dropped when converted to MATLAB sparse
H = GrB.select ('nonzero', G)  % drops the explicit zeros from G
fprintf ('nnz (G): %d  nnz (A): %g nnz (H): %g\n', ...
    nnz (G), nnz (A), nnz (H)) ;
fprintf ('num entries in G: %d\n', GrB.entries (G)) ;

%% Displaying contents of a GraphBLAS matrix
% Unlike MATLAB, the default is to display just a few entries of a GrB matrix.
% Here are all 100 entries of a 10-by-10 matrix, using a non-default disp(G,3):

%%
G = GrB (rand (10)) ;
% display everything:
disp (G,3)

%%
% That was disp(G,3), so every entry was printed.  It's a little long, so
% the default is not to print everything.

%%
% With the default display (level = 2):
G

%%
% That was disp(G,2) or just display(G), which is what is printed by a
% MATLAB statement that doesn't have a trailing semicolon.  With
% level = 1, disp(G,1) gives just a terse summary:
disp (G,1)

%% Storing a matrix by row or by column
% MATLAB stores its sparse matrices by column, refered to as 'sparse by
% col' in SuiteSparse:GraphBLAS.  In the 'sparse by col' format, each
% column of the matrix is stored as a list of entries, with their value
% and row index.  In the 'sparse by row' format, each row is stored as a
% list of values and their column indices.  GraphBLAS uses both 'by row'
% and 'by col', and the two formats can be intermixed arbitrarily.  In
% its C interface, the default format is 'by row'.  However, for better
% compatibility with MATLAB, the SuiteSparse:GraphBLAS MATLAB interface
% uses 'by col' by default instead. 

%%
rng ('default') ;
GrB.clear ;                      % clear prior GraphBLAS settings
fprintf ('the default format is: %s\n', GrB.format) ;
C = sparse (rand (2))
G = GrB (C)
GrB.format (G)

%%
% Many graph algorithms work better in 'by row' format, with matrices
% stored by row.  For example, it is common to use A(i,j) for the edge
% (i,j), and many graph algorithms need to access the out-adjacencies of
% nodes, which is the row A(i,;) for node i.  If the 'by row' format is
% desired, GrB.format ('by row') tells GraphBLAS to create all subsequent
% matrices in the 'by row' format.  Converting from a MATLAB sparse matrix
% (in standard 'by col' format) takes a little more time (requiring a
% transpose), but subsequent graph algorithms can be faster.

%%
G = GrB (C, 'by row')
fprintf ('the format of G is:    %s\n', GrB.format (G)) ;
H = GrB (C)
fprintf ('the format of H is:    %s\n', GrB.format (H)) ;
err = norm (H-G,1)

%% Hypersparse, sparse, bitmap, and full matrices
% SuiteSparse:GraphBLAS can use four kinds of sparse matrix data
% structures: hypersparse, sparse, bitmap, and full, in both 'by col' and
% 'by row' formats, for a total of eight different combinations.  In the
% 'sparse by col' that MATLAB uses for its sparse matrices, an m-by-n
% matrix A takes O(n+nnz(A)) space.  MATLAB can create huge column
% vectors, but not huge matrices (when n is huge).

clear
huge = 2^48 - 1 ;
C = sparse (huge, 1)    % MATLAB can create a huge-by-1 sparse column
try
    C = sparse (huge, huge)     % but this fails
catch me
    error_expected = me
end

%%
% In a GraphBLAS hypersparse matrix, an m-by-n matrix A takes only
% O(nnz(A)) space.  The difference can be huge if nnz (A) << n.

clear
huge = 2^48 - 1 ;
G = GrB (huge, 1)            % no problem for GraphBLAS
H = GrB (huge, huge)         % this works in GraphBLAS too

%%
% Operations on huge hypersparse matrices are very fast; no component of
% the time or space complexity is Omega(n).

I = randperm (huge, 2) ;
J = randperm (huge, 2) ;
H (I,J) = magic (2) ;        % add 4 nonzeros to random locations in H
H (I,I) = 10 * [1 2 ; 3 4] ; % so H^2 is not all zero
H = H^2 ;                    % square H
H = (H' * 2) ;               % transpose H and double the entries
K = pi * spones (H) ;
H = H + K                    % add pi to each entry in H

%% numel uses vpa if the matrix is really huge
e1 = numel (G)               % this is huge, but still a flint
e2 = numel (H)               % this is huge^2, which needs vpa
whos e1 e2

%%
% All of these matrices take very little memory space:
whos C G H K

%% The mask and accumulator
% When not used in overloaded operators or built-in functions, many
% GraphBLAS methods of the form GrB.method ( ... ) can optionally use a
% mask and/or an accumulator operator.  If the accumulator is '+' in
% GrB.mxm, for example, then C = C + A*B is computed.  The mask acts much
% like logical indexing in MATLAB.  With a logical mask matrix M,
% C<M>=A*B allows only part of C to be assigned.  If M(i,j) is true, then
% C(i,j) can be modified.  If false, then C(i,j) is not modified.
%
% For example, to set all values in C that are greater than 0.5 to 3:

%%
A = rand (3) 
C = GrB.assign (A, A > 0.5, 3) ;     % in GraphBLAS
C1 = GrB (A) ; C1 (A > .5) = 3       % also in GraphBLAS
C2 = A       ; C2 (A > .5) = 3       % in MATLAB
err = norm (C - C1, 1)
err = norm (C - C2, 1)

%% The descriptor
% Most GraphBLAS functions of the form GrB.method ( ... ) take an optional
% last argument, called the descriptor.  It is a MATLAB struct that can
% modify the computations performed by the method.  'help
% GrB.descriptorinfo' gives all the details.  The following is a short
% summary of the primary settings:
%
% d.out  = 'default' or 'replace', clears C after the accum op is used.
%
% d.mask = 'default' or 'complement', to use M or ~M as the mask matrix;
%          'structural', or 'structural complement', to use the pattern
%           of M or ~M.
%
% d.in0  = 'default' or 'transpose', to transpose A for C=A*B, C=A+B, etc.
%
% d.in1  = 'default' or 'transpose', to transpose B for C=A*B, C=A+B, etc.
%
% d.kind = 'default', 'GrB', 'sparse', or 'full'; the output of GrB.method.

A = sparse (rand (2)) ;
B = sparse (rand (2)) ;
C1 = A'*B ;
C2 = GrB.mxm ('+.*', A, B, struct ('in0', 'transpose')) ;
err = norm (C1-C2,1)

%% Integer arithmetic is different in GraphBLAS
% MATLAB supports integer arithmetic on its full matrices, using int8,
% int16, int32, int64, uint8, uint16, uint32, or uint64 data types.  None
% of these integer data types can be used to construct a MATLAB sparse
% matrix, which can only be double, double complex, or logical.
% Furthermore, C=A*B is not defined for integer types in MATLAB, except
% when A and/or B are scalars.
%
% GraphBLAS supports all of those types for all of its matrices (hyper,
% sparse, bitmap, or full).  All operations are supported, including C=A*B
% when A or B are any integer type, in 1000s of semirings.
%
% However, integer arithmetic differs in GraphBLAS and MATLAB.  In MATLAB,
% integer values saturate if they exceed their maximum value.  In
% GraphBLAS, integer operators act in a modular fashion.  The latter is
% essential when computing C=A*B over a semiring.  A saturating integer
% operator cannot be used as a monoid since it is not associative.

%%
C = uint8 (magic (3)) ;
G = GrB (C) ;
C1 = C * 40
C2 = G * uint8 (40)
S = double (C1 < 255) ;
assert (isequal (double (C1).*S, double (C2).*S))

%% Example graph algorithm: Luby's method in GraphBLAS
% The GrB.mis function is variant of Luby's randomized algorithm [Luby
% 1985].  It is a parallel method for finding an maximal independent set
% of nodes, where no two nodes are adjacent.  See the GraphBLAS/@GrB/mis.m
% function for details.  The graph must be symmetric with a zero-free
% diagonal, so A is symmetrized first and any diagonal entries are removed.

A = GrB (A) ;
A = GrB.offdiag (A|A') ;

tic
s = GrB.mis (A) ;
toc
fprintf ('# nodes in the graph: %g\n', size (A,1)) ;
fprintf ('# edges: : %g\n', GrB.entries (A) / 2) ;
fprintf ('size of maximal independent set found: %g\n', ...
    full (double (sum (s)))) ;

% make sure it's independent
p = find (s) ;
S = A (p,p) ;
assert (GrB.entries (S) == 0)

% make sure it's maximal
notp = find (s == 0) ;
S = A (notp, p) ;
deg = GrB.vreduce ('+.int64', S) ;
assert (logical (all (deg > 0)))

%% Sparse deep neural network
% The 2019 MIT GraphChallenge (see http://graphchallenge.org) is to solve
% a set of large sparse deep neural network problems.  In this demo, the
% MATLAB reference solution is compared with a solution using GraphBLAS,
% for a randomly constructed neural network.  See the GrB.dnn and
% dnn_builtin.m functions for details.

clear
rng ('default') ;
nlayers = 16 ;
nneurons = 4096 ;
nfeatures = 30000 ;
fprintf ('# layers:   %d\n', nlayers) ;
fprintf ('# neurons:  %d\n', nneurons) ;
fprintf ('# features: %d\n', nfeatures) ;
fprintf ('# of threads used: %d\n', GrB.threads) ;

tic
Y0 = sprand (nfeatures, nneurons, 0.1) ;
for layer = 1:nlayers
    W {layer} = sprand (nneurons, nneurons, 0.01) * 0.2 ;
    bias {layer} = -0.2 * ones (1, nneurons) ;
end
t_setup = toc ;
fprintf ('construct problem time: %g sec\n', t_setup) ;

% convert the problem from MATLAB to GraphBLAS
t = tic ;
[W_gb, bias_gb, Y0_gb] = dnn_builtin2gb (W, bias, Y0) ;
t = toc (t) ;
fprintf ('setup time: %g sec\n', t) ;

%% Solving the sparse deep neural network problem with GraphbLAS
% Please wait ...

tic
Y1 = GrB.dnn (W_gb, bias_gb, Y0_gb) ;
gb_time = toc ;
fprintf ('total time in GraphBLAS: %g sec\n', gb_time) ;

%% Solving the sparse deep neural network problem with MATLAB
% Please wait ...

tic
Y2 = dnn_builtin (W, bias, Y0) ;
builtin_time = toc ;
fprintf ('total time in %s:    %g sec\n', demo_whoami, builtin_time) ;
fprintf ('Speedup of GraphBLAS over %s: %g\n', ...
    demo_whoami, builtin_time / gb_time) ;
fprintf ('\n# of threads used by GraphBLAS: %d\n', GrB.threads) ;

err = norm (Y1-Y2,1)

%% For objects, GraphBLAS has better colon notation than MATLAB
% The MATLAB notation C = A (start:inc:fini) is very handy, and
% it works great if A is a MATLAB matrix.  But for objects like
% the GraphBLAS matrix, MATLAB starts by creating the explicit
% index vector I = start:inc:fini.  That's fine if the matrix is
% modest in size, but GraphBLAS can construct huge matrices.
% The problem is that 1:n cannot be explicitly constructed when n
% is huge.
%
% The C API for GraphBLAS can represent the colon notation 
% start:inc:fini in an implicit manner, so it can do the indexing
% without actually forming the explicit list I = start:inc:fini.
% But there is no access to this method using the MATLAB notation
% start:inc:fini.
%
% Thus, to compute C = A (start:inc:fini) for very huge matrices,
% you need to use use a cell array to represent the colon notation,
% as { start, inc, fini }, instead of start:inc:fini. See
% 'help GrB.extract', 'help GrB.assign' for the functional form.
% For the overloaded syntax C(I,J)=A and C=A(I,J), see
% 'help GrB/subsasgn' and 'help GrB/subsref'.  The cell array
% syntax isn't conventional, but it is far faster than the MATLAB
% colon notation for objects, and takes far less memory when I is huge.

%%
n = 1e14 ;
H = GrB (n, n) ;            % a huge empty matrix
I = [1 1e9 1e12 1e14] ;
M = magic (4)
H (I,I) = M ;
J = {1, 1e13} ;            % represents 1:1e13 colon notation
C1 = H (J, J)              % computes C1 = H (1:e13,1:1e13)
c = nonzeros (C1) ;
m = nonzeros (M (1:3, 1:3)) ;
assert (isequal (c, m)) ;

%%
try
    % try to compute the same thing with colon
    % notation (1:1e13), but this fails:
    C2 = H (1:1e13, 1:1e13)
catch me
    error_expected = me
end

%% Iterative solvers work as-is
% Many built-in functions work with GraphBLAS matrices unmodified.

if (~demo_octave)
    % Octave7: gmres does not yet work for @GrB input matrices
    A = sparse (rand (4)) ;
    b = sparse (rand (4,1)) ;
    x = gmres (A,b)
    norm (A*x-b)
    x = gmres (GrB(A), GrB(b))
    norm (A*x-b)
end

%% ... even in single precision
if (~demo_octave)
    % Octave7: gmres does not yet work for @GrB input matrices
    x = gmres (GrB(A,'single'), GrB(b,'single'))
    norm (A*x-b)
end

%%
% Both of the following uses of minres (A,b) fail to converge because A
% is not symmetric, as the method requires.  Both failures are correctly
% reported, and both the MATLAB version and the GraphBLAS version return
% the same incorrect vector x.

if (exist ('minres'))
    x = minres (A, b)
    x = minres (GrB(A), GrB(b))
end

%%
% With a proper symmetric matrix

if (exist ('minres'))
    A = A+A' ;
    x = minres (A, b)
    norm (A*x-b)
    x = minres (GrB(A), GrB(b))
    norm (A*x-b)
end

%% Extreme performance differences between GraphBLAS and MATLAB.
% The GraphBLAS operations used so far are perhaps 2x to 50x faster than
% the corresponding MATLAB operations, depending on how many cores your
% computer has.  To run a demo illustrating a 500x or more speedup versus
% MATLAB, run this demo:
%
%    gbdemo2
%
% It will illustrate an assignment C(I,J)=A that can take under a second
% in GraphBLAS but several minutes in MATLAB.  To make the comparsion
% even more dramatic, try:
%
%    gbdemo2 (20000)
%
% assuming you have enough memory.

%% Sparse logical indexing is much, much faster in GraphBLAS
% The mask in GraphBLAS acts much like logical indexing in MATLAB, but it
% is not quite the same.  MATLAB logical indexing takes the form:
%
%       C (M) = A (M)
%
% which computes the same thing as the GraphBLAS statement:
%
%       C = GrB.assign (C, M, A)
%
% The GrB.assign statement computes C(M)=A(M), and it is vastly faster
% than C(M)=A(M) for MATLAB sparse matrices, even if the time to convert
% the GrB matrix back to a MATLAB sparse matrix is included.
%
% GraphBLAS can also compute C(M)=A(M) using overloaded operators for
% subsref and subsasgn, but C = GrB.assign (C, M, A) is a bit faster.
%
% Here are both methods in GraphBLAS (both are very fast).  Setting up:

clear
n = 4000 ;
tic
C = sprand (n, n, 0.1) ;
A = 100 * sprand (n, n, 0.1) ;
M = (C > 0.5) ;
t_setup = toc ;
fprintf ('nnz(C): %g, nnz(M): %g, nnz(A): %g\n', ...
    nnz(C), nnz(M), nnz(A)) ;
fprintf ('\nsetup time:     %g sec\n', t_setup) ;

%% First method in GraphBLAS, with GrB.assign
% Including the time to convert C1 from a GraphBLAS
% matrix to a MATLAB sparse matrix:
tic
C1 = GrB.assign (C, M, A) ;
C1 = double (C1) ;
gb_time = toc ;
fprintf ('\nGraphBLAS time: %g sec for GrB.assign\n', gb_time) ;

%% Second method in GraphBLAS, with C(M)=A(M)
% now using overloaded operators, also include the time to
% convert back to a MATLAB sparse matrix, for good measure:
A2 = GrB (A) ;
C2 = GrB (C) ;
tic
C2 (M) = A2 (M) ;
C2 = double (C2) ;
gb_time2 = toc ;
fprintf ('\nGraphBLAS time: %g sec for C(M)=A(M)\n', gb_time2) ;

%% Now with MATLAB matrices, with C(M)=A(M) 
% Please wait, this will take about 10 minutes or so ...

tic
C (M) = A (M) ;
builtin_time = toc ;

fprintf ('\nGraphBLAS time: %g sec (GrB.assign)\n', gb_time) ;
fprintf ('GraphBLAS time: %g sec (overloading)\n', gb_time2) ;
fprintf ('MATLAB time:    %g sec\n', builtin_time) ;
fprintf ('Speedup of GraphBLAS (overloading) over %s: %g\n', ...
    demo_whoami, builtin_time / gb_time2) ;
fprintf ('Speedup of GraphBLAS (GrB.assign)  over %s: %g\n', ...
    demo_whoami, builtin_time / gb_time) ;
fprintf ('\n# of threads used by GraphBLAS: %d\n', GrB.threads) ;

assert (isequal (C1, C))
assert (isequal (C2, C))
fprintf ('Results of GrB and %s match perfectly.\n', demo_whoami)

%% Even more extreme performance differences for C(M)=A(M)
%
% See the tmask script in this GraphBLAS/demo folder for a test comparing
% C(M)=A(M) for a sequence of large matrices.  The test takes far to long
% to run in this demo, because MATLAB is exceedingly slow.  Below are
% the results on a 4-core Dell laptop:
%
%         n      nnz(C)     nnz(M)  GraphBLAS  MATLAB    speedup
%                                   (sec)      (sec)
%     2,048      20,432      2,048  0.005         0.024     4.7
%     4,096      40,908      4,096  0.003         0.115     39
%     8,192      81,876      8,191  0.009         0.594     68
%    16,384     163,789     16,384  0.009         2.53     273
%    32,768     327,633     32,767  0.014        12.4      864
%    65,536     655,309     65,536  0.025        65.9    2,617
%   131,072   1,310,677    131,070  0.055       276.2    4,986
%   262,144   2,621,396    262,142  0.071     1,077.    15,172
%   524,288   5,242,830    524,288  0.114     5,855.    51,274
% 1,048,576  10,485,713  1,048,576  0.197    27,196.   137,776
% 2,097,152  20,971,475  2,097,152  0.406   100,799.   248,200
% 4,194,304  41,942,995  4,194,304  0.855   ~4 days?   500,000?
%
% The last run for MATLAB never finished but I estimate it would
% take about 4 to 5 days.

%% Limitations and their future solutions
% The MATLAB interface for SuiteSparse:GraphBLAS is a work-in-progress.
% It has some limitations, most of which will be resolved over time.
%
% (1) Nonblocking mode:
%
% GraphBLAS has a 'non-blocking' mode, in which operations can be left
% pending and completed later.  SuiteSparse:GraphBLAS uses the
% non-blocking mode to speed up a sequence of assignment operations, such
% as C(I,J)=A.  However, in its MATLAB interface, this would require a
% MATLAB mexFunction to modify its inputs.  That breaks the MATLAB API
% standard, so it cannot be safely done.  As a result, using GraphBLAS
% via its MATLAB interface can be slower than when using its C API.

%%
% (2) Integer element-wise operations:
%
% Integer operations in MATLAB saturate, so that uint8(255)+1 is 255.  To
% allow for integer monoids, GraphBLAS uses modular arithmetic instead.
% This is the only way that C=A*B can be defined for integer semirings.
% However, saturating integer operators could be added in the future, so
% that element- wise integer operations on GraphBLAS sparse integer
% matrices could work just the same as their MATLAB counterparts.
%
% So in the future, you could perhaps write this, for both sparse and
% dense integer matrices A and B:
%
%       C = GrB.eadd ('+saturate.int8', A, B)
%
% to compute the same thing as C=A+B in MATLAB for its full int8
% matrices.  Note that MATLAB can do this only for dense integer
% matrices, since it doesn't support sparse integer matrices.

%%
% (3) Faster methods:
%
% Most methods in this MATLAB interface are based on efficient parallel C
% functions in GraphBLAS itself, and are typically as fast or faster than
% the equivalent built-in operators and functions in MATLAB.
%
% There are few notable exceptions; these will be addressed in the future.
% These include reshape, issymmetric, and ishermitian, all of which should
% be faster in a future release.

%%
% (4) Linear indexing:
%
% If A is an m-by-n 2D MATLAB matrix, with n > 1, A(:) is a column vector
% of length m*n.  The index operation A(i) accesses the ith entry in the
% vector A(:).  This is called linear indexing in MATLAB.  It is not yet
% fully available for GraphBLAS matrices in this MATLAB interface to
% GraphBLAS, but will be added in the future.

%%
% (5) Implicit singleton dimension expansion 
%
% In MATLAB C=A+B where A is m-by-n and B is a 1-by-n row vector
% implicitly expands B to a matrix, computing C(i,j)=A(i,j)+B(j).  This
% implicit expansion is not yet supported in GraphBLAS with C=A+B.
% However, it can be done with C = GrB.mxm ('+.+', A, diag(GrB(B))).
% That's a nice example of the power of semirings, but it's not
% immediately obvious, and not as clear a syntax as C=A+B.  The
% GraphBLAS/@GrB/dnn.m function uses this 'plus.plus' semiring to
% apply the bias to each neuron.

A = magic (3)
B = 1000:1000:3000
C1 = A + B
C2 = GrB.mxm ('+.+', A, diag (GrB (B)))
err = norm (C1-C2,1)

%%
% (6) MATLAB object overhead.
%
% The GrB matrix is a MATLAB object, and there are some cases where
% performance issues can arise as a result.  Extracting the contents of
% a MATLAB object (G.field) takes much more time than for a MATLAB struct
% with % the same syntax, and building an object has similar issues.  The
% difference is small, and it does not affect large problems.  But if you
% have many calls to GrB operations with a small amount of work, then the
% time can be dominated by the MATLAB object-oriented overhead.
%
% There is no solution or workaround to this issue.

A = rand (3,4) ;
G = GrB (A) ;
tic
for k = 1:100000
    [m, n] = size (A) ;
end
toc
tic
for k = 1:100000
    [m, n] = size (G) ;
end
toc

%% GraphBLAS operations
% In addition to the overloaded operators (such as C=A*B) and overloaded
% functions (such as L=tril(A)), GraphBLAS also has methods of the form
% GrB.method.  Most of them take an optional input matrix Cin, which is
% the initial value of the matrix C for the expression below, an optional
% mask matrix M, and an optional accumulator operator.
%
%   in GrB syntax:  C<#M,replace> = accum (C, A*B)
%
%   in @GrB: C = GrB.mxm (Cin, M, accum, semiring, A, B, desc) ;
%
% In the above expression, #M is either empty (no mask), M (with a mask
% matrix) or ~M (with a complemented mask matrix), as determined by the
% descriptor (desc).  'replace' can be used to clear C after it is used
% in accum(C,T) but before it is assigned with C<...> = Z, where
% Z=accum(C,T).  The matrix T is the result of some operation, such as
% T=A*B for GrB.mxm, or T=op(A,B) for GrB.eadd.
%
% For a complete list of GraphBLAS overloaded operators and methods, type:
%
%   help GrB
%
% Thanks for watching!
%
% Tim Davis, Texas A&M University, http://faculty.cse.tamu.edu/davis,
% https://twitter.com/DocSparse

