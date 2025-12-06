classdef GrB
%GrB GraphBLAS sparse matrices for Octave/MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse
% linear algebraic operations over semirings.  Visit http://graphblas.org
% for more details and resources.  See also the SuiteSparse:GraphBLAS User
% Guide in this package.
%
% The GrB class represents a GraphBLAS sparse matrix.  The GrB
% method creates a GraphBLAS sparse matrix from a built-in matrix.  Other
% methods also generate GrB matrices.  For example:
%
%   G = GrB.subassign (C, M, A) ;
%
% constructs a GraphBLAS matrix G, which is the result of C<M>=A in
% GraphBLAS notation (like C(M)=A(M)).  The matrices used in any
% GrB.method may be built-in matrices (sparse or full) or GraphBLAS
% matrices (hyper, sparse, bitmap, or full, by row or column), in any
% combination.
%
% --------------------
% The GrB constructor:
% --------------------
%
%   The GrB constructor creates a GraphBLAS matrix.  The input A may be
%   any built-in or GraphBLAS matrix:
%
%   C = GrB (A) ;            GraphBLAS copy of a matrix A, same type
%   C = GrB (m, n) ;         m-by-n GraphBLAS double matrix, no entries
%   C = GrB (..., type) ;    create or typecast to a different type
%   C = GrB (..., format) ;  create in a specified format
%
%   The m and n parameters above are built-in scalars.  The type and format
%   parameters are strings.  The default format is 'by col', to match the
%   format used in built-in (see also GrB.format), but many graph
%   algorithms are faster if the format is 'by row'.  The format can also
%   specify the data structure to use (hypersparse, sparse, bitmap, and/or
%   full).
%
%   The usage C = GrB (m, n, type) is analgous to A = sparse (m, n), which
%   creates an empty built-in sparse matrix A.  The type parameter is a
%   string, which defaults to 'double' if not present.
%
%   For the usage C = GrB (A, type), A is either a built-in sparse or full
%   matrix, or a GraphBLAS sparse matrix object.  C is created as a
%   GraphBLAS sparse matrix object that contains a copy of A, typecasted
%   to the given type if the type string does not match the type of A.
%   If the type string is not present it defaults to 'double'.
%
% --------------------
% Matrix types:
% --------------------
%
%   Most of the valid type strings correspond to built-in class of the
%   same name (see 'help class'):
%
%       'logical'           8-bit boolean
%       'int8'              8-bit signed integer
%       'int16'             16-bit signed integer
%       'int32'             32-bit signed integer
%       'int64'             64-bit signed integer
%       'uint8'             8-bit unsigned integer
%       'uint16'            16-bit unsigned integer
%       'uint32'            32-bit unsigned integer
%       'uint64'            64-bit unsigned integer
%       'double'            64-bit floating-point (real, not complex)
%       'single'            32-bit floating-point (real, not complex)
%       'single complex'    single complex
%       'double complex'    double complex (also just 'complex')
%
%   In built-in matrices, complex is an attribute, not a class.  In GrB
%   matrices, 'double complex' and 'single complex' are treated as their
%   own data types.
%
% ---------------
% Matrix formats:
% ---------------
%
%   The format of a GraphBLAS matrix can have a large impact on
%   performance.  GraphBLAS matrices can be stored by column or by row.
%   The corresponding format string is 'by col' or 'by row', respectively.
%   Since the only format for built-in sparse and full matrices is 'by
%   col', that is the default format for GraphBLAS matrices via this
%   interface to GraphBLAS.  However, the default for the C API is 'by
%   row' since graph algorithms tend to be faster with that format.
%
%   Column vectors are always stored 'by col', and row vectors are always
%   stored 'by row'.  The format for new matrices propagates from the
%   format of their inputs.  For example with C=A*B, C takes on the same
%   format as A, unless A is a vector, in which case C takes on the format
%   of B.  If both A and B are vectors, then the format of C is determined
%   by the descriptor (if present), or by the default format (see
%   GrB.format).
%
%   When a GraphBLAS matrix is converted into a built-in sparse or full
%   matrix, it is always returned as 'by col'.
%
%   The format can also specify the data structure to use.  By default
%   GraphBLAS selects automatically between hypersparse, sparse, bitmap,
%   and full formats.  See 'help GrB.format' for details.
%
%--------------------
% Integer operations:
%--------------------
%
%   Operations on integer values differ from built-in operations, where
%   uint8(255)+1 is 255, since the arithmetic saturates.  This is not
%   possible in matrix operations such as C=A*B, since saturation of
%   integer arithmetic would render most of the monoids useless.
%   GraphBLAS instead computes a result modulo the word size, so that
%   GrB(uint8(255))+1 is zero.  However, new unary and binary operators
%   could be added so that element-wise operations saturate.  The C
%   interface allows for arbitrary creation of user-defined operators, so
%   this could be added in the future.  See 'help GrB/MATLAB_vs_GrB' for
%   more details.
%
%-------------------------------------------------------------------------
% Methods for the GrB class:
%-------------------------------------------------------------------------
%
%   C = GrB (...)           construct a GraphBLAS matrix
%
% Overloaded operators (all except 'colon'):
%
%   C = and (A, B)          C = A & B
%   C = ctranspose (G)      C = G'
%   i = end (G, k, ndims)   A(1:end,1:end)
%   C = eq (A, B)           C = A == B
%   C = ge (A, B)           C = A >= B
%   C = gt (A, B)           C = A > B
%   C = horzcat (A, B)      C = [A , B]
%   C = ldivide (A, B)      C = A .\ B
%   C = le (A, B)           C = A <= B
%   C = lt (A, B)           C = A < B
%   C = minus (A, B)        C = A - B
%   C = mldivide (A, B)     C = A \ B
%   C = mpower (A, B)       C = A ^ B
%   C = mrdivide (A, B)     C = A / B
%   C = mtimes (A, B)       C = A * B
%   C = ne (A, B)           C = A ~= B
%   C = not (G)             C = ~G
%   C = or (A, B)           C = A | B
%   C = plus (A, B)         C = A + B
%   C = power (A, B)        C = A .^ B
%   C = rdivide (A, B)      C = A ./ B
%   I = subsindex (G)       X = A (G)
%   C = subsasgn (C, S, A)  C (I,J) = A or C (M) = A
%   C = subsref (A, S)      C = A (I,J) or C = A (M)
%   C = times (A, B)        C = A .* B
%   C = transpose (G)       C = G.'
%   C = uminus (G)          C = -G
%   C = uplus (G)           C = +G
%   C = vertcat (A, B)      C = [A ; B]
%
% Overloaded functions:
%
%   C = abs (G)             absolute value
%   C = acos (G)            inverse cosine
%   C = acosh (G)           inverse hyperbolic cosine
%   C = acot (G)            inverse cotangent
%   C = acoth (G)           inverse hyperbolic cotangent
%   C = acsc (G)            inverse cosecant
%   C = acsch (G)           inverse hyperbolic cosecant
%   C = all (G, ...)        reduce via '&', to vector or scalar
%   p = amd (G, ...)        approximate minimum degree ordering
%   C = angle (G)           phase angle of a complex matrix
%   C = any (G, ...)        reduce via '|', to vector or scalar
%   C = asec (G)            inverse secant
%   C = asech (G)           inverse hyperbolic secant
%   C = asin (G)            inverse sine
%   C = asinh (G)           inverse hyperbolic sine
%   assert (G)              generate an error if G is false
%   C = atan (G)            inverse tangent
%   C = atanh (G)           inverse hyperbolic tangent
%   C = atan2 (A, B)        inverse tangent (four-quadrant)
%
%   [lo, hi] = bandwidth (G, ...)   lower and upper bandwidth of G
%   C = bitand (A, B, ...)          bitwise and
%   C = bitcmp (A, ...)             bitwise negation
%   C = bitget (A, B, ...)          get bits
%   C = bitset (A, B, ...)          set bits
%   C = bitshift (A, B, ...)        shift bits
%   C = bitor (A, B, ...)           bitwise or
%   C = bitxor (A, B, ...)          bitwise xor
%
%   C = cast (G, ...)       cast GrB matrix to built-in matrix
%   C = cat (dim, ...)      contatenate matrices
%   C = ceil (G)            round towards infinity
%   C = cell2mat (A)        concatenate a cell array of matrices
%   p = colamd (G)          column approximate minimum degree ordering
%   C = complex (G)         cast GrB matrix to built-in sparse complex
%   C = conj (G)            complex conjugate
%   C = cos (G)             cosine
%   C = cosh (G)            hyperbolic cosine
%   C = cot (G)             cotangent
%   C = coth (G)            hyperbolic cotangent
%   C = csc (G)             cosecant
%   C = csch (G)            hyperbolic cosecant
%   C = cbrt (G)            cube root
%
%   C = diag (A, k)         diagonal matrices and diagonals
%   DiGraph = digraph (G,...)   directed Graph
%   disp (A, level)         display a built-in or GrB matrix A
%   display (G)             display a GrB matrix G; same as disp(G,2)
%   [...] = dmperm (G)      Dulmage-Mendelsohn permutation
%   C = double (G)          cast GrB matrix to built-in sparse double
%
%   [V, ...] = eig (G,...)  eigenvalues and eigenvectors
%   G = GrB.empty (m, n)    empty matrix for the GrB class
%   C = eps (G)             floating-point spacing
%   C = erf (G)             error function
%   C = erfc (G)            complementary error function
%   p = etree (G)           elimination tree
%   C = exp (G)             natural exponent
%   C = expm1 (G)           exp (x) - 1
%
%   [I,J,X] = find (G, ...) extract entries from a matrix
%   C = fix (G)             round towards zero
%   C = flip (G, dim)       flip the order of entries
%   C = floor (G)           round towards -infinity
%   c = fprintf (...)       print to a file or to the Command Window
%   C = full (G, ...)       adds explicit zeros or id values
%
%   C = gamma (G)           gamma function
%   C = gammaln (G)         logarithm of gamma function
%   Graph = graph (G, ...)  undirected graph
%
%   C = hypot (A, B)        sqrt of sum of squares
%
%   C = imag (G)            imaginary part of a complex matrix
%   C = int8 (G)            cast GrB matrix to built-in full int8
%   C = int16 (G)           cast GrB matrix to built-in full int16
%   C = int32 (G)           cast GrB matrix to built-in full int32
%   C = int64 (G)           cast GrB matrix to built-in full int64
%   s = isa (G, classname)  check if a GrB matrix is of a specific class
%   s = isbanded (G,...)    true if G is banded
%   s = iscolumn (G)        true if n=1, for an m-by-n GrB matrix G
%   s = isdiag (G)          true if G is diagonal
%   s = isempty (G)         true if any dimension of G is zero
%   s = isequal (A, B)      test if equal
%   C = isfinite (G)        test if finite
%   s = isfloat (G)         true if GrB matrix is double, single, complex
%   s = ishermitian (G)     true if G is Hermitian
%   C = isinf (G)           test if infinite
%   s = isinteger (G)       true if GrB matrix is int8, int16, ..., uint64
%   s = islogical (G)       true if GrB matrix is logical
%   s = ismatrix (G)        true for any GrB matrix G
%   C = isnan (G)           test if NaN
%   s = isnumeric (G)       true for any GrB matrix G (even logical)
%   s = isreal (G)          true if GrB matrix is not complex
%   s = isrow (G)           true if m=1, for an m-by-n GrB matrix G
%   s = isscalar (G)        true if G is a 1-by-1 GrB matrix
%   s = issparse (G)        true for any GrB matrix G
%   s = issymmetric (G)     true if G is symmetric
%   s = istril (G)          true if G is lower triangular
%   s = istriu (G)          true if G is upper triangular
%   s = isvector (G)        true if m=1 or n=1, for an m-by-n GrB matrix G
%
%   C = kron (A, B)         Kronecker product
%
%   n = length (G)          length of a GrB vector
%   C = log (G)             natural logarithm
%   C = log10 (G)           base-10 logarithm
%   C = log1p (G)           log (1+x)
%   [F, E] = log2 (G)       base-2 logarithm
%   C = logical (G)         cast GrB matrix to built-in sparse logical
%
%   C = mat2cell (A,m,n)    break a matrix into a cell array of matrices
%   C = max (A,B,option)    reduce via max, to vector or scalar
%   C = min (A,B,option)    reduce via min, to vector or scalar
%
%   e = nnz (G)             number of entries in a GrB matrix G
%   X = nonzeros (G)        extract all entries from a GrB matrix
%   s = norm (G, kind)      norm of a GrB matrix
%   C = num2cell (A,dim)    convert a matrix into a cell array
%   e = numel (G)           m*n for an m-by-n GrB matrix G
%   e = nzmax (G)           number of entries in a GrB matrix G
%
%
%   C = pow2 (F, E)         base-2 power
%   C = prod (G, option)    reduce via product, to vector or scalar
%
%   C = real (G)            real part of a complex matrix
%   C = repmat (G, ...)     replicate and tile a GraphBLAS matrix
%   C = reshape (G, ...)    reshape a GraphBLAS matrix
%   C = round (G)           round towards nearest
%
%   C = sec (G)             secant
%   C = sech (G)            hyperbolic secant
%   C = sign (G)            signum function
%   C = sin (G)             sine
%   C = single (G)          cast GrB matrix to built-in full single
%   C = sinh (G)            hyperbolic sine
%   [m,n,t] = size (G,dim)  size and type of a GrB matrix
%   C = sparse (G)          makes a copy of a GrB matrix
%   C = spfun (fun, G)      evaluate a function on the entries of G
%   C = spones (G, type)    return pattern of GrB matrix
%   C = sprand (...)        random GraphBLAS matrix
%   C = sprandn (...)       random GraphBLAS matrix, normal distribution
%   C = sprandsym (...)     random symmetric GraphBLAS matrix
%   c = sprintf (...)       print to a string
%   C = sqrt (G)            element-wise square root
%   C = sum (G, option)     reduce via sum, to vector or scalar
%   p = symamd (G)          approximate minimum degree ordering
%   p = symrcm (G)          reverse Cuthill-McKee ordering
%
%   C = tan (G)             tangent
%   C = tanh (G)            hyperbolic tangent
%   L = tril (G, k)         lower triangular part of GrB matrix G
%   U = triu (G, k)         upper triangular part of GrB matrix G
%
%   C = uint8 (G)           cast GrB matrix to built-in full uint8
%   C = uint16 (G)          cast GrB matrix to built-in full uint16
%   C = uint32 (G)          cast GrB matrix to built-in full uint32
%   C = uint64 (G)          cast GrB matrix to built-in full uint64
%
%   C = xor (A, B)          exclusive or
%
%-------------------------------------------------------------------------
% Static Methods:
%-------------------------------------------------------------------------
%
%   The Static Methods for the GrB class can be used on input matrices of
%   any kind: GraphBLAS sparse matrices, built-in sparse matrices, or
%   built-in full matrices, in any combination.  The output matrix C is a
%   GraphBLAS matrix, by default, but can be optionally returned as a
%   built-in sparse or full matrix.  The static methods divide into three
%   categories: those that perform basic functions, graph algorithms, and
%   the 12 foundational GraphBLAS operations.
%
%---------------------------
% GraphBLAS basic functions:
%---------------------------
%
%   context:
%   GrB.clear                    clear GraphBLAS workspace and settings
%   GrB.finalize                 finish GraphBLAS
%   GrB.init                     initialize GraphBLAS
%   t = GrB.threads (t)          set/get # of threads to use in GraphBLAS
%   c = GrB.chunk (c)            set/get chunk size to use in GraphBLAS
%   b = GrB.burble (b)           set/get burble (diagnostic output)
%   [s,path] = GrB.jit (s,path)  control the GraphBLAS JIT
%
%   info:
%   GrB.binopinfo (op, type)     list properties of a binary operator
%   list = GrB.binops            list all binary ops
%   GrB.descriptorinfo (d)       list properties of a descriptor
%   GrB.monoidinfo (op, type)    list properties of a monoid
%   list = GrB.monoids           list all monoids
%   GrB.selectopinfo (op, type)  list properties of a select operator
%   list = GrB.selectops         list all selectops
%   GrB.semiringinfo (s, type)   list properties of a semiring
%   list = GrB.semirings         list all semirings
%   GrB.unopinfo (op, type)      list properties of a unary operator
%   list = GrB.unops             list all unary ops
%
%   basic matrices:
%   C = GrB.false (...)          all-false logical matrix
%   C = GrB.true (...)           all-true logical matrix
%   C = GrB.ones (...)           matrix with all ones
%   C = GrB.zeros (...)          all-zero matrix
%
%   operations:
%   C = GrB.build (I,J,X,m,n,dup,type,desc) build a GrB matrix from
%                                list of entries (like C=sparse(I,J,X...))
%   [C,I,J] = GrB.compact (A,id,s) remove empty rows and columns
%   c = GrB.entries (A,...)      count or query entries in a matrix
%   C = GrB.expand (scalar, A)   expand a scalar (C = scalar*spones(A))
%   [I,J,X] = GrB.extracttuples (A,desc) extract all entries (like 'find')
%   C = GrB.eye (m,n,type)       identity matrix of any type (like 'speye')
%   f = GrB.format (f)           set/get matrix format by row or col
%   s = GrB.isbyrow (A)          true if format f A is 'by row'
%   s = GrB.isbycol (A)          true if format f A is 'by col'
%   s = GrB.isfull (A)           true if all entries present
%   s = GrB.issigned (type)      true if type is signed
%   c = GrB.nonz (A,...)         count or query nonzeros in a matrix
%   s = GrB.normdiff (A,B,kind)  norm (A-B,kind)
%   C = GrB.offdiag (A)          prune diagonal entries
%   C = GrB.prune (A, id)        prune entries equal to id
%   C = GrB.random (...)         random GraphBLAS matrix (like 'sprand')
%   C = GrB.speye (m,n,type)     identity matrix of any type (like 'speye')
%   t = GrB.type (A)             get the type of a built-in or GrB matrix A
%   v = GrB.version              string with SuiteSparse:GraphBLAS version
%   v = GrB.ver                  struct with SuiteSparse:GraphBLAS version
%
%   load/save:
%   C = GrB.load (filename)      load a single matrix from a file
%   GrB.save (A, filename)       save a single matrix to a file
%
%-------------------------------------
% Static Methods for graph algorithms:
%-------------------------------------
%
%   [v, parent] = GrB.bfs (A, s, ...) ;     breadth-first search
%   Y = GrB.dnn (W, bias, Y0) ;             deep neural network
%   C = GrB.incidence (A, ...) ;            incidence matrix
%   C = GrB.ktruss (A, k, check) ;          k-truss
%   L = GrB.laplacian (A, type, check) ;    Laplacian graph
%   iset = GrB.mis (A, check) ;             maximal independent set
%   r = GrB.pagerank (A, opts) ;            PageRank of a matrix
%   s = GrB.tricount (A, check) ;           triangle count
%
%-----------------------------------
% Foundational GraphBLAS operations:
%-----------------------------------
%
%   GraphBLAS has 12 foundational operations, listed below.  All have
%   similar parameters.  The full set of input parameters is listed in the
%   order in which they appear in the GraphBLAS C API, except that for the
%   @GrB interface, Cin and C are different matrices.  They combine into a
%   single input/output matrix in the GraphBLAS C API.  In the @GrB
%   interface, many of the parameters become optional, and they can appear
%   in different order.
%
%   GrB.apply       apply a unary operator
%   GrB.apply2      apply a binary operator
%   GrB.assign      sparse matrix assignment, such as C(I,J)=A
%   GrB.eadd        element-wise addition
%   GrB.eunion      element-wise union
%   GrB.emult       element-wise multiplication
%   GrB.extract     extract submatrix, like C=A(I,J)
%   GrB.kronecker   Kronecker product
%   GrB.mxm         sparse matrix-matrix multiplication over a semiring
%   GrB.reduce      reduce a matrix to a scalar
%   GrB.select      select a subset of entries from a matrix
%   GrB.subassign   sparse matrix assignment, such as C(I,J)=A
%   GrB.trans       transpose a matrix
%   GrB.vreduce     reduce a matrix to a vector
%
%   In GraphBLAS notation (with C, Cin arguments for the one matrix
%   C), these take the following form:
%
%       C<#M,replace> = accum (C, operation (A or A', B or B'))
%
%   C is both an input and output matrix.  In this interface to
%   GraphBLAS, it is split into Cin (the value of C on input) and C
%   (the value of C on output).  M is the optional mask matrix, and #M is
%   either M or ~M depending on whether or not the mask is complemented
%   via the desc.mask option.  The replace option is determined by
%   desc.out; if present, C is cleared after it is used in the accum
%   operation but before the final assignment.  A and/or B may optionally
%   be transposed via the descriptor fields desc.in0 and desc.in1,
%   respectively.  To select the format of C, use desc.format.  See
%   GrB.descriptorinfo for more details.
%
%   accum is optional; if not is not present, then the operation becomes
%   C<...> = operation(A,B).  Otherwise, C = C + operation(A,B) is
%   computed where '+' is the accum operator.  It acts like a sparse
%   matrix addition (see GrB.eadd), in terms of the structure of the
%   result C, but any binary operator can be used.
%
%   The mask M acts like built-in logical indexing.  If M(i,j)=1 then
%   C(i,j) can be modified; if zero, it cannot be modified by the
%   operation.
%
%   The full list of parameters is shown below:
%
%       C = GrB.apply     (Cin, M, accum, op, A,          desc)
%       C = GrB.apply2    (Cin, M, accum, op, A, B,       desc)
%       C = GrB.assign    (Cin, M, accum,     A,    I, J, desc)
%       C = GrB.eadd      (Cin, M, accum, op, A, B,       desc)
%       C = GrB.eunion    (Cin, M, accum, op, A, a, B, b, desc)
%       C = GrB.emult     (Cin, M, accum, op, A, B,       desc)
%       C = GrB.extract   (Cin, M, accum,     A,    I, J, desc)
%       C = GrB.kronecker (Cin, M, accum, op, A, B,       desc)
%       C = GrB.mxm       (Cin, M, accum, op, A, B,       desc)
%       C = GrB.reduce    (Cin,    accum, op, A,          desc)
%       C = GrB.select    (Cin, M, accum, op, A, b,       desc)
%       C = GrB.subassign (Cin, M, accum,     A,    I, J, desc)
%       C = GrB.trans     (Cin, M, accum,     A,          desc)
%       C = GrB.vreduce   (Cin, M, accum, op, A,          desc)
%
%   The parameters divide into 4 classes: matrices, strings, cells, and a
%   single optional struct (the descriptor).  The order of parameters
%   between the matrices, strings, and cell classes is arbitrary.  The
%   order of parameters within a class is important; for example, if a
%   method takes 4 matrix inputs, then they must appear in the order Cin,
%   M, A, and then B.  However, if a single string appears as a
%   parameter, it can appear anywhere within the list of 4 matrices.
%
%   (1) Cin, M, A, B are matrices, and a and b are scalars (eunion only).
%       If the method takes up to 4 matrices
%       (mxm, kronecker, select (with operator requiring a b
%       parameter), eadd, emult, apply2), then they appear in this order:
%       with 2 matrix inputs: A, B
%       with 3 matrix inputs: Cin, A, B
%       with 4 matrix inputs: Cin, M, A, B
%       For GrB.select, b is a scalar.  For GrB.apply2, either A or B
%       is a scalar.
%
%       If the method takes up to 3 matrices (vreduce, apply, assign,
%       subassign, extract, trans, or select without b):
%       with 1 matrix input:  A
%       with 2 matrix inputs: Cin, A
%       with 3 matrix inputs: Cin, M, A
%       Note that assign and subassign require Cin.
%
%       If the method takes up to 2 input matrices (the reduce method):
%       with 1 matrix input:  A
%       with 2 matrix inputs: Cin, A
%
%   (2) accum and op are strings.  The accum string is always optional.
%       If the method has an op parameter, then it is a required input.
%
%       If the method has both parameters, and just one string appears,
%       it is the op, which is a semiring for mxm, a unary operator for
%       apply, a select operator for the select method, and a binary
%       operator for all other methods.  If 2 strings appear, the first
%       one is the accum the second is the op.  If the accum appears then
%       Cin must also appear as a matrix input.
%
%       If the method has no op (assign, subassign, extract, trans), but
%       just an accum parameter, then 0 or 1 strings may appear in the
%       parameter list.  If a string appears, it is the accum.
%
%   (3) I and J are cell arrays.  For details, see the assign, subassign,
%       and extract methods; a short summary appears below.  Both are
%       optional:
%       with no cell inputs: default for I and J
%       with 1  cell inputs: I, default for J
%       with 2  cell inputs: I, J
%
%       Each cell array may appear with 0, 1, 2, or 3 items:
%           0: { }                  ":" in built-in notation
%           1: { list }             a list of integer indices
%           2: { start,fini }       start:fini in built-in notation
%           3: { start,inc,fini }   start:inc:fini in built-in notation
%
%   (4) The descriptor is an optional struct.  If present, it must
%       appear last, after all other parameters.
%
%   Some valid uses are shown below, along with their equivalent in
%   GraphBLAS notation.  For the first three mxm examples, the four
%   matrices C, M, A, and B must appear in that order, and the two
%   strings '+' and '+.*' must appear in that order, but the matrices and
%   strings may be interleaved arbitrarily.
%
%       C = GrB.apply (C, M, '|', '~', A)           C<M> |= ~A
%       C = GrB.apply ('~', A)                      C = ~A
%
%       C = GrB.assign (C, M, '+', A, I, J)         C(I,J)<M> += A
%       C = GrB.assign (C, I, J, M, '+', A)         C(I,J)<M> += A
%
%       C = GrB.assign (C, A, I, J)                 C(I,J) = A
%       C = GrB.assign (C, I, J, A)                 C(I,J) = A
%       C = GrB.assign (C, A)                       C = A
%       C = GrB.assign (C, M, A)                    C<M> = A
%       C = GrB.assign (C, M, '+', A)               C<M> += A
%       C = GrB.assign (C, '+', A, I)               C (I,:) += A
%
%       C = GrB.emult (C, M, '+', A, '*', B)        C<M> += A.*B
%       C = GrB.emult (A, '*', B)                   C = A.*B
%
%       C = GrB.extract (C, M, '+', A, I, J)        C<M> += A(I,J)
%       C = GrB.extract (A, I, J)                   C = A(I,J)
%       C = GrB.extract (I, J, A)                   C = A(I,J)
%       C = GrB.extract (A)                         C = A
%       C = GrB.extract (C, M, A)                   C<M> = A
%       C = GrB.extract (C, M, '+', A)              C<M> += A
%       C = GrB.extract (C, '+', A, I)              C += A(I,:)
%
%       C = GrB.mxm (C, M, '+', '+.*', A, B)        C<M> += A*B
%       C = GrB.mxm (C, M, '+', A, '+.*', B)        C<M> += A*B
%       C = GrB.mxm ('+', '+,*', C, M, A, B)        C<M> += A*B
%
%       C = GrB.mxm ('+.*', A, B)                   C = A*B
%       C = GrB.mxm (A, '+.*', B)                   C = A*B
%       C = GrB.mxm (C, M, A, '+.*', B)             C<M> = A*B
%
%       c = GrB.reduce (c, '+', 'max', A)           c += max (A)
%       c = GrB.reduce ('max', A)                   c = max (A)
%       c = GrB.reduce (A, 'max')                   c = max (A)
%       c = GrB.reduce (c, 'max', A)                c = max (A)
%
% See also sparse.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

properties (SetAccess = private, GetAccess = private)
    % The struct contains the entire opaque content of a GraphBLAS
    % GrB_Matrix.
    opaque = [ ] ;
end

methods

    %---------------------------------------------------------------------
    % GrB: GraphBLAS matrix constructor
    %---------------------------------------------------------------------

    function C = GrB (arg1, arg2, arg3, arg4)
    %GRB GraphBLAS constructor: create a GraphBLAS sparse matrix.
    %
    % C = GrB (A) ;          GrB copy of a matrix A, same type and format
    %
    % C = GrB (A, type) ;    GrB typecasted copy of a matrix A, same format
    % C = GrB (A, format) ;  GrB copy of a matrix A, with given format
    % C = GrB (m, n) ;       empty m-by-n GrB double matrix, default format
    %
    % C = GrB (A, type, format) ;   GrB copy of A, new type and format
    % C = GrB (A, format, type) ;   ditto
    %
    % C = GrB (m,n, type) ;   empty m-by-n GrB type matrix, default format
    % C = GrB (m,n, format) ; empty m-by-n GrB double matrix, given format
    %
    % C = GrB (m,n,type,format) ;  empty m-by-n matrix, given type & format
    % C = GrB (m,n,format,type) ;  ditto
    %
    % See also sparse.
        if (nargin == 1)
            if (isstruct (arg1))
                % C = GrB (A), where the input A is a GraphBLAS struct as
                % returned by another GrB* function, but this usage is not
                % meant for the end-user.  It is only used internally in
                % @GrB, to convert a GraphBLAS struct computed by a
                % GraphBLAS mexFunction into a GrB matrix object.
                C.opaque = arg1 ;
            elseif (isobject (arg1))
                % arg1 is already a GrB matrix; make a deep copy
                C.opaque = gbnew (arg1.opaque) ;
            else
                % arg1 is a built-in matrix; convert to a GrB matrix
                C.opaque = gbnew (arg1) ;
            end
        else
            if (isobject (arg1))
                % extract the contents of the GrB object as its opaque
                % struct so the gbnew mexFunction can access it.
                arg1 = arg1.opaque ;
            end
            % varargin is more elegant, but it is slower than the switch
            switch (nargin)
                case 2
                    C.opaque = gbnew (arg1, arg2) ;
                case 3
                    C.opaque = gbnew (arg1, arg2, arg3) ;
                case 4
                    C.opaque = gbnew (arg1, arg2, arg3, arg4) ;
            end
        end
    end

    %---------------------------------------------------------------------
    % implicitly-defined methods
    %---------------------------------------------------------------------

    % The following methods work without any implemention needed here,
    % because they are built-in m-files that can operate with GrB
    % inputs:
    %
    %   matrix operations: flipdim fliplr flipud cast isrow iscolumn ndims
    %   sprank etreeplot spy gplot reallog realpow realsqrt
    %
    %   iterative solvers: bicgstabl bicgstab cgs minres gmres bicg pcg
    %   qmr rjr tfqmr lsqr

    %---------------------------------------------------------------------
    % FUTURE:: many these could also be overloaded:
    %---------------------------------------------------------------------

    % methods in the ops folder:
    %
    %   colon idivide ismembertol uniquetol
    %   m-files: intersect ismember setdiff setxorunion unique
    %       (if 'sort' is overloaded, and 1D indexing added,
    %       then all these will work for GrB matrices)

    % methods in the datatypes folder:
    %
    %   typecast swapbytes

    % methods in the datafun folder:
    %
    %   cummax cummin cumprod cumsum diff histcounts islocalmax
    %   ismissing issorted maxk mink movmad movmax movmean movmedian
    %   movmin movprod movstd movsum movvar rmmissing rmoutliers
    %   sort sortrows standardizeMissing topkrows
    %
    %   m-files: bounds corrcoef cov del2 fillmissing filloutliers
    %   gradient isoutlier issortedrows mean median mode normalize
    %   rescale smoothdata std var

    % methods the 'double' class that are not yet implemented here:
    %
    %   Possible 'double' functions to overload in the future (note that
    %   mod and rem are not the same as the ANSI fmod or remainder, but
    %   the built-in rem is almost the same as the ANSI fmod):
    %
    %       mod rem unwrap sind asind cosd acosd tand
    %       atand secd asecd cscd acscd cotd acotd atan2d
    %
    %   not needed:
    %
    %       special functions: airy bernoulli besselh besseli besselj
    %       besselk bessely betainc betaincinv chebyshevT chebyshevU
    %       coshint cosint dawson dilog dirac ei ellipticCE ellipticCK
    %       ellipticCPi ellipticE ellipticF ellipticK ellipticNome
    %       ellipticPi erfcinv erfcx erfi erfinv fresnelc fresnels
    %       gammainc gammaincinv harmonic igamma jacobiP kummerU laguerreL
    %       legendreP logint pochhammer psi signIm sinhint sinint ssinint
    %       whittakerM whittakerW wrightOmega zeta triangularPulse
    %       rectangularPulse
    %
    %       eigenvalue-related: charpoly euler gegenbauerC hermiteH jordan
    %       minpoly poly2sym polylog
    %
    %       others: colon factor divisors superiorfloat

    % methods in matfun not implemented here:
    %
    %       balance cdf2rdf chol cholupdate condeig condest cond
    %       decomposition det expm funm gsvd hess inv ldl linsolve logm
    %       lscov lsqminnorm ltitr lu normest1 normest null ordeig ordqz
    %       ordschur orth pinv planerot polyeig qrdelete qrinsert qr
    %       qrupdate qz rank rcond rref rsf2csf schur sqrtm svd sylvester
    %       trace vecnorm

    % methods in sparfun not implemented here:
    %
    %       colperm delsq dissect eigs ichol ilu spalloc spaugment
    %       spconvert spdiags svds symbfact symmlq unmesh
    %
    %       not needed: treeplot treelayout numgrid nested spparms

    % methods in elmat not implemented here:
    %
    %       accumarray blkdiag bsxfun circshift compan gallery
    %       hadamard hankel hilb inf invhilb ipermute isequaln nan ndgrid
    %       pascal permute repelem rot90 shiftdim toeplitz vander
    %       wilkinson
    %
    %       not needed: linspace logspace ind2sub sub2ind meshgrid pi
    %       freqspace flintmax intmax intmin squeeze realmin realmax i j
    %       magic rosser

    % methods for classes graph and digraph not yet implemented:
    %
    %       addedge addnode bfsearch centrality conncomp dfsearch
    %       distances findedge findnode isisomorphic isomorphism maxflow
    %       nearest outedges rmedge rmnode shortestpath shortestpathtree
    %       simplify

    % methods for class graph (not in digraph class) not yet implemented:
    %
    %       bctree biconncomp minspantree neighbors

    % methods for class digraph (not in graph class) not yet implemented:
    %
    %       condensation inedges isdag predecessors successors toposort
    %       transclosure transreduction

    % methods in LAGraph: (see the LAGraph/src folder)

    %---------------------------------------------------------------------
    % operator overloading
    %---------------------------------------------------------------------

    C = and (A, B) ;            % C = (A & B)
    C = ctranspose (A) ;        % C = A'
    i = end (A, k, ndims) ;     % for A (1:end,1:end)
    C = eq (A, B) ;             % C = (A == B)
    C = ge (A, B) ;             % C = (A >= B)
    C = gt (A, B) ;             % C = (A > B)
    C = horzcat (varargin) ;    % C = [A , B]
    C = ldivide (A, B) ;        % C = A .\ B
    C = le (A, B) ;             % C = (A <= B)
    C = lt (A, B) ;             % C = (A < B)
    C = minus (A, B) ;          % C = A - B
    C = mldivide (A, B) ;       % C = A \ B
    C = mpower (A, B) ;         % C = A^B
    C = mrdivide (A, B) ;       % C = A / B
    C = mtimes (A, B) ;         % C = A * B
    C = ne (A, B) ;             % C = (A ~= B)
    C = not (G) ;               % C = ~A
    C = or (A, B) ;             % C = (A | B)
    C = plus (A, B) ;           % C = A + B
    C = power (A, B) ;          % C = A .^ B
    C = rdivide (A, B) ;        % C = A ./ B
    I = subsindex (A) ;         % for C = X (A), using A as index I
    C = subsasgn (C, S, A) ;    % C (I,J) = A or C (M) = A
    C = subsref (A, S) ;        % C = A (I,J) or C = A (M)
    C = times (A, B) ;          % C = A .* B
    C = transpose (G) ;         % C = A.'
    C = uminus (G) ;            % C = -A
    C = uplus (G) ;             % C = +A
    C = vertcat (varargin) ;    % C = [A ; B]

    %---------------------------------------------------------------------
    % Methods that overload built-in functions:
    %---------------------------------------------------------------------

    %   In the list below, G is always a GraphBLAS matrix.  The inputs A
    %   and B can be a mix of GraphBLAS and/or built-in matrices, but at
    %   least one will be a GraphBLAS matrix because these are all methods
    %   that are overloaded from the built-in versions.  If all inputs are
    %   built-in matrices, these methods are not used.  The output matrix
    %   (C, L, or U) is always a GraphBLAS matrix.  Lower case variables
    %   i, e, s, and n are scalars.  Outputs p, parent, I, J, and X are
    %   built-in vectors.  Graph is a built-in undirected graph.  DiGraph
    %   is a built-in directed digraph.

    C = abs (G) ;
    C = acos (G) ;
    C = acosh (G) ;
    C = acot (G) ;
    C = acoth (G) ;
    C = acsc (G) ;
    C = acsch (G) ;
    C = all (G, option) ;
    p = amd (G, varargin) ;
    C = angle (G) ;
    C = any (G, option) ;
    C = asec (G) ;
    C = asech (G) ;
    C = asin (G) ;
    C = asinh (G) ;
    assert (G) ;            % test assertion 
    C = atan (G) ;
    C = atanh (G) ;
    C = atan2 (A, B) ;

    [lo, hi] = bandwidth (G, uplo) ;
    C = bitand (A, B, assumedtype) ;
    C = bitcmp (A, assumedtype) ;
    C = bitget (A, B, assumedtype) ;
    C = bitset (A, B, arg3, arg4) ;
    C = bitshift (A, B, arg3) ;
    C = bitor (A, B, assumedtype) ;
    C = bitxor (A, B, assumedtype) ;

%   C = cast (G, ...)       built-in works as-is
    C = cat (dim, varargin) ;
    C = ceil (G) ;
    [p, varargout] = colamd (G, varargin) ;
    C = complex (A, B) ;
    C = conj (G) ;
    C = cos (G) ;
    C = cosh (G) ;
    C = cot (G) ;
    C = coth (G) ;
    C = csc (G) ;
    C = csch (G) ;
    C = cbrt (G) ;

    C = diag (A, k) ;
    DiGraph = digraph (G, option) ;
    disp (A, level) ;
    display (G) ;
    [p, varargout] = dmperm (G) ;
    C = double (G) ;

    [V, varargout] = eig (G, varargin) ;        % uses GrB matrices
    C = eps (G) ;
    C = erf (G) ;
    C = erfc (G) ;
    [parent, varargout] = etree (G, varargin) ;
    C = exp (G) ;
    C = expm1 (G) ;

    [I,J,X] = find (G, k, search) ;
    C = fix (G) ;
    C = flip (G, dim) ;
    C = floor (G) ;
    c = fprintf (varargin) ;
    C = full (A, type, identity) ;

    C = gamma (G) ;
    C = gammaln (G) ;
    Graph = graph (G, varargin) ;               % uses GrB matrices

    C = hypot (A, B) ;

    C = imag (G) ;
    C = int8 (G) ;
    C = int16 (G) ;
    C = int32 (G) ;
    C = int64 (G) ;
    s = isa (G, type) ;
    s = isbanded (G, lo, hi) ;
%   s = iscolumn (G)        built-in works as-is
    s = isdiag (G) ;
    s = isempty (G) ;
    s = isequal (A, B) ;
    C = isfinite (G) ;
    s = isfloat (G) ;
    s = ishermitian (G, option) ;
    C = isinf (G) ;
    s = isinteger (G) ;
    s = islogical (G) ;
    s = ismatrix (G) ;
    C = isnan (G) ;
    s = isnumeric (G) ;
    s = isreal (G) ;
%   s = isrow (G)           built-in works as-is
    s = isscalar (G) ;
    s = issparse (G) ;
    s = issymmetric (G, option) ;
    s = istril (G) ;
    s = istriu (G) ;
    s = isvector (G) ;

    C = kron (A, B) ;

    n = length (G) ;
    C = log (G) ;
    C = log10 (G) ;
    C = log1p (G) ;
    [F, E] = log2 (G) ;
    C = logical (G) ;

    C = mat2cell (A, m, n) ;
    C = max (A, B, option) ;
    C = min (A, B, option) ;

    e = nnz (G) ;
    X = nonzeros (G) ;
    s = norm (G, kind) ;
    C = num2cell (A, dim) ;
    s = numel (G) ;
    e = nzmax (G) ;

    C = pow2 (A, B) ;
    C = prod (G, option) ;

    C = real (G) ;
    C = repmat (G, m, n) ;
    C = reshape (G, m, n, by_col) ;
    C = round (G) ;

    C = sec (G) ;
    C = sech (G) ;
    C = sign (G) ;
    C = sin (G) ;
    C = single (G) ;
    C = sinh (G) ;
    [m, n, t] = size (G, dim) ;
    C = sparse (G) ;
    C = spfun (fun, G) ;
    C = spones (G, type) ;
    C = sprand (arg1, arg2, arg3) ;
    C = sprandn (arg1, arg2, arg3) ;
    C = sprandsym (arg1, arg2) ;
    c = sprintf (varargin) ;
    C = sqrt (G) ;
    S = struct (G) ;
    C = sum (G, option) ;
    [p, varargout] = symamd (G, varargin) ;
    p = symrcm (G) ;

    C = tan (G) ;
    C = tanh (G) ;
    L = tril (G, k) ;
    U = triu (G, k) ;

    C = uint8 (G) ;
    C = uint16 (G) ;
    C = uint32 (G) ;
    C = uint64 (G) ;

    C = xor (A, B) ;

end

methods (Static)

    %---------------------------------------------------------------------
    % Static Methods:
    %---------------------------------------------------------------------

    % All of these are used as GrB.method (...), with the "GrB." prefix.
    % The input matrices (A, B, C, M, ...) are of any kind (GraphBLAS,
    % built-in sparse, or built-in full).  The output matrix C is a
    % GraphBLAS matrix.

    % Some of the methods listed below are high-level graph algorithms that
    % rely on GrB objects internally (bfs, dnn, ktruss, mis, pagerank, and
    % tricount), for simplicity and readability.  All of the other methods
    % extract the opaque content of the GrB objects just once, operate on
    % them, and then cast their results back into a GrB object just
    % once.  This makes for less-readable code, but it avoids the
    % performance cost of accessing/modifying a object.

    MATLAB_vs_GrB ;
    C = apply (Cin, M, accum, op, A, desc) ;
    C = apply2 (Cin, M, accum, op, A, B, desc) ;
    [x,p] = argmin (A, dim) ;
    [C,P] = argsort (A, dim, direction) ;
    [x,p] = argmax (A, dim) ;
    C = assign (Cin, M, accum, A, I, J, desc) ;
    [v, parent] = bfs (A, s, varargin) ;        % uses GrB matrices
    binopinfo (op, type) ;
    list = binops ;
    C = build (I, J, X, m, n, dup, type, desc) ;
    b = burble (b) ;
    C = cell2mat (A) ;
    c = chunk (c) ;
    clear ;
    [C, I, J] = compact (A, id, symmetric) ;
    descriptorinfo (d) ;
    C = deserialize (blob, mode, arg3) ;        % arg3 for testing only
    Y = dnn (W, bias, Y0) ;                     % uses GrB matrices
    C = eadd (Cin, M, accum, op, A, B, desc) ;
    C = empty (arg1, arg2) ;
    C = emult (Cin, M, accum, op, A, B, desc) ;
    x = entries (A, arg2, arg3) ;
    C = expand (scalar, A, type) ;
    C = extract (Cin, M, accum, A, I, J, desc) ;
    [I, J, X] = extracttuples (A, desc) ;
    C = eunion (Cin, M, accum, op, A, a, B, b, desc) ;
    C = eye (m, n, type) ;
    finalize ;
    [f, s, iso] = format (arg) ;
    C = incidence (A, varargin) ;
    init ;
    s = isbyrow (A) ;
    s = isbycol (A) ;
    s = isfull (A) ;
    s = issigned (arg) ;
    C = kronecker (Cin, M, accum, op, A, B, desc) ;
    C = ktruss (A, k, check) ;                  % uses GrB matrices
    L = laplacian (A, type, check) ;
    C = load (filename) ;
    iset = mis (A, check) ;                     % uses GrB matrices
    monoidinfo (monoid, type) ;
    list = monoids ;
    C = mxm (Cin, M, accum, semiring, A, B, desc) ;
    result = nonz (A, varargin) ;
    s = normdiff (A, B, kind) ;
    C = offdiag (A) ;
    ctype = optype (a, b) ;
    [r, stats] = pagerank (A, opts) ;           % uses GrB matrices
    C = prune (A, identity) ;
    C = random (varargin) ;
    C = reduce (cin, accum, monoid, A, desc) ;
    filename_used = save (C, filename) ;
    C = select (Cin, M, accum, selectop, A, b, desc) ;
    selectopinfo (op, type) ;
    list = selectops ;
    semiringinfo (s, type) ;
    list = semirings ;
    blob = serialize (A, method, level) ;
    C = speye (m, n, type) ;
    C = subassign (Cin, M, accum, A, I, J, desc) ;
    nthreads = threads (nthreads) ;
    [s,path] = jit (s,path) ;
    C = trans (Cin, M, accum, A, desc) ;
    s = tricount (A, check, d) ;                % uses GrB matrices
    s = type (A) ;
    unopinfo (op, type) ;
    list = unops ;
    v = version ;
    v = ver ;
    C = vreduce (Cin, M, accum, monoid, A, desc) ;

    % these were formerly overloaded methods, now Static methods
    C = false (varargin) ;
    C = true (varargin) ;
    C = ones (varargin) ;
    C = zeros (varargin) ;

end
end

