%% Operations on built-in matrices vs GraphBLAS matrices
%
% Most of the overloaded operations on GrB matrices work just the same as
% the built-in operations of the same name.  There are some important
% differences.  In future versions, the GrB interface to GraphBLAS
% may be modified to minimize these differences.
%
% ------------------------------------------------
%% Matrix classes and types:
% ------------------------------------------------
%
%     Octave/MATLAB supports 3 kinds of sparse matrices: logical, double,
%     and double complex.  For single precision floating-point (real or
%     complex), and integer matrices, Octave/MATLAB only supports full
%     matrices, not sparse.
%
%     GraphBLAS supports all types:  logical, int8, int16, int32, int64,
%     uint8, uint16, uint32, uint64, single, double, single complex, and
%     double complex.  GraphBLAS has only a single class: the GrB object.
%     It uses a 'type' to represent these different data types.  See help
%     GrB.type for more details.
%
% ------------------------------------------------
%% Explicit zeros:
% ------------------------------------------------
%
%     Octave/MATLAB always drops explicit zeros from its sparse matrices.
%     GraphBLAS never drops them, except on request (A = GrB.prune (A)).
%     This difference will always exist between Octave/MATLAB and
%     GraphBLAS.
%
%     GraphBLAS cannot drop zeros automatically, since the explicit zero
%     might be meaningful.  The value zero is the additive identity for
%     the single monoid supported by built-in (the '+' of the '+.*'
%     conventional semiring).  Octave/MATLAB has only two semirings
%     ('+.*.double' and '+.*.double complex').  GraphBLAS supports both of
%     those, but many 1000s more, many of which have a different identity
%     value.  In a shortest-path problem, for example, an edge of weight
%     zero is very different than no edge at all (the identity is inf, for
%     the 'min' monoid often used in path problems).
%
% ------------------------------------------------
%% Linear indexing:
% ------------------------------------------------
%
%     In Octave/MATLAB, as in A = rand (3) ; X = A (1:6) extracts the
%     first two columns of A as a 6-by-1 vector.  Except for C=A(:),
%     this is not yet supported in GraphBLAS, but will be added in
%     the future.
%
% ------------------------------------------------
%% Increasing/decreasing the size of a matrix:
% ------------------------------------------------
%
%     This can be done with a built-in matrix, and the result is a sparse
%     10-by-10 sparse matrix A:
%
%         clear A
%         A (1) = sparse (pi)     % A is created as 1-by-1
%         A (10,10) = 42          % A becomes 10-by-10
%         A (5,:) = [ ]           % delete row 5
%
%     The GraphBLAS equivalent does not yet work, since submatrix indexing
%     does not yet increase the size of the matrix:
%
%         clear A
%         A (1) = GrB (pi)            % fails since A does not exist
%         A = GrB (pi)                % works
%         A (10,10) = 42              % fails, since A is 1-by-1
%
%     This feature is not yet supported but may be added in the future.
%
% ------------------------------------------------
%% min and max operations on complex matrices:
% ------------------------------------------------
%
%     Octave/MATLAB can compute the min and max on complex values (they
%     return the entry with the largest magnitude).  This is not
%     well-defined mathematically, since the resulting min and max
%     operations cannot be used as monoids, as they can for real types
%     (integer or floating-point types).  As a result, GraphBLAS does not
%     support min and max for complex types, and will never do so.
%
%     GraphBLAS uses the 'omitnan' behavior, which is the default in
%     Octave/MATLAB.  The 'includenam' option is not available in
%     GraphBLAS, but may appear in the future.
%
%     Likewise, logical comparators (< <= > >=) are not well-defined
%     mathematically for complex types.  Octave/MATLAB defines them, but
%     GraphBLAS does not.  GraphBLAS can only compare for equality (==)
%     and inequality (~=) with complex types.
%
% ------------------------------------------------
%% Singleton expansion:
% ------------------------------------------------
%
%     Octave/MATLAB can expand a 'singleton' dimension (of size 1) of one
%     input to match the required size of the other input.  For example,
%     given
%
%         A = rand (4)
%         x = [10 100 1000 10000]
%
%     these computations both scale the columns of x.  The results are the
%     same:
%
%         A.*x            % singleton expansion
%         A * diag(x)     % standard matrix-vector multiply, which works
%
%     GraphBLAS does not yet support singleton expansion:
%
%         A = GrB (A)
%         A * diag (x)    % works
%         A.*x            % fails
%
% ------------------------------------------------
%% Typecasting from floating-point types to integer:
% ------------------------------------------------
%
%     In Octave/MATLAB, the default is to round to the nearest integer.
%     If the fractional part is exactly 0.5: the integer with larger
%     magnitude is selected.  In GraphBLAS, typecasting matches the
%     built-in behavior when explicitly converting matrices:
%
%       G = 100 * rand (4)
%       G = GrB (G, 'int8')
%
%     However, if a double matrix is used as-is directly in an integer
%     semiring, the C typecasting rules are used:
%
%       % suppose A and B are double:
%       A = 5 * rand (4) ;
%       B = 5 * rand (4) ;
%
%       % uses GraphBLAS typecasting
%       C = GrB.mxm (A, '+.*.int8', B)
%
%       % uses built-in typecasting:
%       C = GrB.mxm (GrB (A, 'int8'), '+.*.int8', GrB (B, 'int8'))
%
% ------------------------------------------------
%% Mixing different integers:
% ------------------------------------------------
%
%     Octave/MATLAB refuses to compute int16(1) + int8(1).  GraphBLAS can
%     do this, using the rules listed by:
%
%         help GrB.optype
%
% ------------------------------------------------
%% Combining 32-bit or lower integers and floating-point:
% ------------------------------------------------
%
%     Both Octave/MATLAB and GraphBLAS do the work in floating-point.  In
%     Octave/MATLAB, the result is then cast to the integer type.  In
%     GraphBLAS, the GrB matrix has the floating-point type.
%     Octave/MATLAB can only do this if the floating-point operand is a
%     scalar; GraphBLAS can work with any matrices of valid sizes.
%
%     To use the Octave/MATLAB rule in GraphBLAS: after computing the
%     result, simply typecast to the desired integer type with
%
%       A = cast (5 * rand (4), 'int8') ;
%       % C is int8:
%       C = A+pi
%       A = GrB (A)
%       % C is double:
%       C = A+pi
%       % C is now int8:
%       C = GrB (C, 'int8')
%
% ------------------------------------------------
%% 64-bit integers (int64 and uint64) and double:
% ------------------------------------------------
%
%     In Octave/MATLAB, both inputs are converted to 80-bit long double
%     (floating-poing) and then the result is typecasted back to the
%     integer type.  In GraphBLAS the work is done in double, and the
%     result is left in the double type.
%
%     This can be done in Octave/MATLAB only if the double operator is a
%     scalar, as in A+pi.  With GraphBLAS, A+B can mix arbitrary types,
%     but A+pi is computed in double, not long double.
%
%     This feature may be added to GraphBLAS in the future, by adding
%     new operators that internally do their work in long double.
%
% ------------------------------------------------
%% Octave/MATLAB integer operations saturate:
% ------------------------------------------------
%
%     If a = uint8 (255), and b = uint8 (1), then a+b for built-in
%     matrices is 255.  That is, the results saturate on overflow or
%     underflow, to the largest and smallest integer respectively.
%
%     This kind of arithmetic is not compatible with integer semirings,
%     and thus Octave/MATLAB does not support integer matrix computations
%     such as C=A*B.
%
%     GraphBLAS supports integer semirings, and to do so requires
%     integer operations that act in a modulo fashion.  As a result if
%     a=GrB(255,'uint8') and b=GrB(1,'uint8'), then a+b is zero.
%
%     It would be possible to add saturating binary operators to replicate
%     the saturating integer behavior in Octave/MATLAB, since this is
%     useful for operations such as A+B or A.*B for signals and images.
%     This may be added in the future, as C = GrB.eadd (A, '+saturate', B)
%     for example.
%
%     This affects the following operators and functions, and likely more
%     as well:
%
%         +   plus
%         -   minus
%         -   uminus (as in C = -A)
%         .*  times
%         ./  ldivide
%         .\  rdivide
%         .^  power
%         sum, prod:  built-in converts to double; GraphBLAS keeps the
%                     type of the input
%
%     It does not affect the following:
%
%         +   uplus
%         *   mtimes (GraphBLAS can do this, built-in method can't)
%         <   lt
%         <=  le
%         >   gt
%         >=  ge
%         ==  eq
%         ~=  ne
%             bitwise operators (bitor, bitand, ...)
%         ~   logical negation
%         |   or
%         &   and
%         '   ctranspose
%         .'  transpose
%             subsref
%             subsasgn
%             end
%
% ------------------------------------------------
%% The rules for concatenation differ.
% ------------------------------------------------
%
%     For C = [A1 A2] and [A1 ; A2], the type of C differs.
%     GraphBLAS uses the rules given by 'help GrB.optype'.
%
% ------------------------------------------------
%% Bitwise operators:
% ------------------------------------------------
%
%     GraphBLAS includes all the bitwise operators that Octave/MATLAB has.
%     In addition, GraphBLAS can use the bitwise operations in semirings;
%     for example, if A and B are uint8, then:
%
%         C = GrB.mxm (A, 'bitor.bitand', B) ;
%
%     computes C = A*B using the 'bitor.bitand.uint8' semiring.  Try:
%
%         GrB.semiringinfo ('bitor.bitand.uint8')
%
%% For more details, see the GraphBLAS user guide in GraphBLAS/Doc.
% See also GrB, sparse.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

help GrB.MATLAB_vs_GrB ;

