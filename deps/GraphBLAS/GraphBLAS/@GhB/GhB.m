classdef (InferiorClasses = {?GrB}) GhB < handle & GrB
%GhB GraphBLAS sparse matrices for Octave/MATLAB.
%
% GraphBLAS is a library for creating graph algorithms based on sparse linear
% algebraic operations over semirings.  Visit http://graphblas.org for more
% details and resources.  See also the SuiteSparse:GraphBLAS User Guide in this
% package.
%
% The GhB matrix is the handle variant of the GrB value matrix object.  It is
% nearly identical to the GrB matrix, with additional syntax for modifying a
% GhB matrix in-place. 
%
% A GrB_Matrix can contain pending work after it is computed, but this work
% must be finished when using a GrB matrix.  The GhB handle matrix object
% allows the work to remain unfinished, to be done later.  This feature results
% in faster computations than when using GrB matrices.
%
% All GrB methods are also available as GhB methods; see "help GrB" for
% details, and just replace "GrB" with "GhB".
%
% GrB and GhB matrices can be mixed.  In MATLAB, if any matrix in a computation
% is a GhB matrix, the result is a GhB matrix.  Octave 11.1 is different; C=A+B
% creates C as GrB if A is GrB and B is GhB.  GhB matrices are only created via
% the GhB(...) constructor.  Thus, if only GrB.* methods are used, all matrices
% will be GrB.
%
% The GhB matrix is a handle object, so C can also be modified in place.  Using
% this in-place syntax, which cannot be done with GrB:
%
%       GhB.apply     (C, M, accum, op, A,          desc)
%       GhB.apply2    (C, M, accum, op, A, B,       desc)
%       GhB.assign    (C, M, accum,     A,    I, J, desc)
%       GhB.eadd      (C, M, accum, op, A, B,       desc)
%       GhB.eunion    (C, M, accum, op, A, a, B, b, desc)
%       GhB.emult     (C, M, accum, op, A, B,       desc)
%       GhB.extract   (C, M, accum,     A,    I, J, desc)
%       GhB.kronecker (C, M, accum, op, A, B,       desc)
%       GhB.mxm       (C, M, accum, op, A, B,       desc)
%       GhB.reduce    (C,    accum, op, A,          desc)
%       GhB.select    (C, M, accum, op, A, b,       desc)
%       GhB.subassign (C, M, accum,     A,    I, J, desc)
%       GhB.trans     (C, M, accum,     A,          desc)
%       GhB.vreduce   (C, M, accum, op, A,          desc)
%
% For the in-place syntax, no output parameter ("C = GhB.method (..)") can
% appear, and the matrix C must appear as an input parameter.
%
% Example in-place usage:
%
%       GhB.apply (C, M, '|', '~', A)           C<M> |= ~A
%
%       GhB.assign (C, M, '+', A, I, J)         C(I,J)<M> += A
%       GhB.assign (C, I, J, M, '+', A)         C(I,J)<M> += A
%
%       GhB.assign (C, A, I, J)                 C(I,J) = A
%       GhB.assign (C, I, J, A)                 C(I,J) = A
%       GhB.assign (C, A)                       C = A
%       GhB.assign (C, M, A)                    C<M> = A
%       GhB.assign (C, M, '+', A)               C<M> += A
%       GhB.assign (C, '+', A, I)               C (I,:) += A
%
%       GhB.emult (C, M, '+', A, '*', B)        C<M> += A.*B
%
%       GhB.extract (C, M, '+', A, I, J)        C<M> += A(I,J)
%       GhB.extract (C, M, A)                   C<M> = A
%       GhB.extract (C, M, '+', A)              C<M> += A
%       GhB.extract (C, '+', A, I)              C += A(I,:)
%
%       GhB.mxm (C, M, '+', '+.*', A, B)        C<M> += A*B
%       GhB.mxm (C, M, '+', A, '+.*', B)        C<M> += A*B
%
%       GhB.mxm (C, M, A, '+.*', B)             C<M> = A*B
%
%       GhB.reduce (c, '+', 'max', A)           c += max (A)
%       GhB.reduce (c, 'max', A)                c = max (A)
%
% See also GrB, sparse.
%
% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

methods

    %---------------------------------------------------------------------------
    % GhB: GraphBLAS matrix constructor
    %---------------------------------------------------------------------------

    function C = GhB (arg1, arg2, arg3, arg4)
    %GHB GraphBLAS constructor: create a GraphBLAS GhB handle matrix.
    %
    % C = GhB (A) ;          GhB copy of a matrix A, same type and format
    %
    % C = GhB (A, type) ;    GhB typecasted copy of A, same format
    % C = GhB (A, format) ;  GhB copy of a matrix A, with given format
    % C = GhB (m, n) ;       empty m-by-n GhB double matrix
    %
    % C = GhB (A, type, format) ; GhB copy of A, new type and format
    % C = GhB (A, format, type) ; ditto
    %
    % C = GhB (m,n, type) ;   empty m-by-n GhB type matrix, default format
    % C = GhB (m,n, format) ; empty m-by-n GhB double matrix, given format
    %
    % C = GhB (m,n,type,format) ; empty m-by-n matrix, given type & format
    % C = GhB (m,n,format,type) ; ditto
    %
    % See also sparse.
        already_struct = (nargin >= 1 && isstruct (arg1)) ;
        if (nargin >= 1 && gb_is_grb (arg1))
            arg1 = struct (arg1) ;
        end
        switch (nargin)
            case 0
                C.opaque = [ ] ;
            case 1
                if (already_struct)
                    C.opaque = arg1 ;
                else
                    C.opaque = gbmex_new (1, arg1) ;
                end
            case 2
                C.opaque = gbmex_new (1, arg1, arg2) ;
            case 3
                C.opaque = gbmex_new (1, arg1, arg2, arg3) ;
            case 4
                C.opaque = gbmex_new (1, arg1, arg2, arg3, arg4) ;
        end
    end

    %---------------------------------------------------------------------------
    % GraphBLAS GhB destructor
    %---------------------------------------------------------------------------

    function delete (C)
    %DELETE delete a GhB matrix
    gbmex_delete (C) ;
    end

    %---------------------------------------------------------------------------
    % operator overloading
    %---------------------------------------------------------------------------

    C = and (A, B) ;            % C = (A & B)
    C = ctranspose (A) ;        % C = A'
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
    C = subsasgn (C, S, A) ;    % C (I,J) = A or C (M) = A
    C = subsref (A, S) ;        % C = A (I,J) or C = A (M)
    C = times (A, B) ;          % C = A .* B
    C = transpose (G) ;         % C = A.'
    C = uminus (G) ;            % C = -A
    C = uplus (G) ;             % C = +A
    C = vertcat (varargin) ;    % C = [A ; B]

    %---------------------------------------------------------------------------
    % Methods that overload built-in functions:
    %---------------------------------------------------------------------------

    C = abs (G) ;
    C = acos (G) ;
    C = acosh (G) ;
    C = acot (G) ;
    C = acoth (G) ;
    C = acsc (G) ;
    C = acsch (G) ;
    C = all (G, option) ;
    C = angle (G) ;
    C = any (G, option) ;
    C = asec (G) ;
    C = asech (G) ;
    C = asin (G) ;
    C = asinh (G) ;
    C = atan (G) ;
    C = atanh (G) ;
    C = atan2 (A, B) ;
    C = bitand (A, B, assumedtype) ;
    C = bitcmp (A, assumedtype) ;
    C = bitget (A, B, assumedtype) ;
    C = bitor (A, B, assumedtype) ;
    C = bitset (A, B, arg3, arg4) ;
    C = bitshift (A, B, arg3) ;
    C = bitxor (A, B, assumedtype) ;
    C = cat (dim, varargin) ;
    C = cbrt (G) ;
    C = ceil (G) ;
    C = conj (G) ;
    C = cos (G) ;
    C = cosh (G) ;
    C = cot (G) ;
    C = coth (G) ;
    C = csc (G) ;
    C = csch (G) ;
    C = diag (A, k) ;
    C = eps (G) ;
    C = erf (G) ;
    C = erfc (G) ;
    C = exp (G) ;
    C = expm1 (G) ;
    C = fix (G) ;
    C = flip (G, dim) ;
    C = floor (G) ;
    C = full (A, type, identity) ;
    C = gamma (G) ;
    C = gammaln (G) ;
    C = hypot (A, B) ;
    C = imag (G) ;
    C = isfinite (G) ;
    C = isinf (G) ;
    C = isnan (G) ;
    C = kron (A, B) ;
    C = log (G) ;
    C = log10 (G) ;
    C = log1p (G) ;
    [F, E] = log2 (G) ;
    C = mat2cell (A, m, n) ;
    C = max (A, B, option) ;
    C = min (A, B, option) ;
    C = num2cell (A, dim) ;
    C = pow2 (A, B) ;
    C = prod (G, option) ;
    C = real (G) ;
    C = repmat (G, m, n) ;
    C = reshape (G, m, n, by_col) ;
    C = round (G) ;
    S = saveobj (G) ;
    C = sec (G) ;
    C = sech (G) ;
    C = sign (G) ;
    C = sin (G) ;
    C = sinh (G) ;
    C = sparse (G) ;
    C = spfun (fun, G) ;
    C = spones (G, type) ;
    C = sprand (varargin) ;
    C = sprandn (varargin) ;
    C = sprandsym (varargin) ;
    C = sqrt (G) ;
    C = sum (G, option) ;
    C = tan (G) ;
    C = tanh (G) ;
    C = tril (G, k) ;
    C = triu (G, k) ;
    C = xor (A, B) ;

end

methods (Static)

    %---------------------------------------------------------------------------
    % Static Methods:
    %---------------------------------------------------------------------------

    % All of these are used as GhB.method (...) with the GhB prefix.  The input
    % matrices (Cin, M, A, B, M, ...) are of any kind, except for the 14
    % foundational methods when used with in-place syntax (where Cin must be
    % GhB), and GhB.set.  The outputs are GhB matrices except where noted.

    % the 14 foundational methods:
    C = apply (Cin, M, accum, op, A, desc) ;
    C = apply2 (Cin, M, accum, op, A, B, desc) ;
    C = assign (Cin, M, accum, A, I, J, desc) ;
    C = eadd (Cin, M, accum, op, A, B, desc) ;
    C = emult (Cin, M, accum, op, A, B, desc) ;
    C = eunion (Cin, M, accum, op, A, a, B, b, desc) ;
    C = extract (Cin, M, accum, A, I, J, desc) ;
    C = kronecker (Cin, M, accum, op, A, B, desc) ;
    C = mxm (Cin, M, accum, semiring, A, B, desc) ;
    C = reduce (cin, accum, monoid, A, desc) ;
    C = select (Cin, M, accum, selectop, A, b, desc) ;
    C = subassign (Cin, M, accum, A, I, J, desc) ;
    C = trans (Cin, M, accum, A, desc) ;
    C = vreduce (Cin, M, accum, monoid, A, desc) ;

    [x, p] = argmax (A, dim) ;
    [x, p] = argmin (A, dim) ;
    [C, P] = argsort (A, dim, direction) ;
    [v, parent] = bfs (A, s, varargin) ;
    C = build (I, J, X, m, n, dup, type, desc) ;
    C = cell2mat (A) ;
    [C, I, J] = compact (A, id, symmetric) ;
    C = deserialize (blob) ;
    Y = dnn (W, bias, Y0) ;
    C = empty (arg1, arg2) ;
    x = entries (A, arg2, arg3) ;           % returns a built-in x
    C = expand (scalar, A, type) ;
    C = eye (m, n, type) ;
    C = false (varargin) ;
    C = incidence (A, varargin) ;
    C = ktruss (A, k, symmetric) ;
    L = laplacian (A, type, check) ;
    C = load (filename) ;
    C = loadobj (S) ;
    iset = mis (A, check) ;
    result = nonz (A, varargin) ;           % returns a built-in or GhB result
    C = offdiag (A) ;
    C = ones (varargin) ;
    [r, stats] = pagerank (A, opts) ;
    C = prune (A, identity) ;
    C = random (varargin) ;
    blob = serialize (A, method, level) ;   % returns a built-in blob
    C = speye (m, n, type) ;
    s = tricount (A, check, d) ;            % returns a built-in scalar
    C = true (varargin) ;
    C = zeros (varargin) ;

    % these appear in GhB only, not GrB.
    value = get (G, state) ;                % G can be GhB, GrB, or built-in
    set (G, state, value) ;                 % modifies G in place; G must be GhB

end
end

