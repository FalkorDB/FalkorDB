function C = gb_random (varargin)
%GB_RANDOM uniformly distributed random GraphBLAS matrix.
% Implements C = GrB.random (...), C = sprand (...), C = sprand (...),

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%---------------------------------------------------------------------------
% parse inputs
%---------------------------------------------------------------------------

% defaults
dist = 'uniform' ;
type = 'double' ;
range = [ ] ;
sym_option = 'unsymmetric' ;
firstchar = nargin + 1 ;

% look for strings
for k = 1:nargin
    arg = varargin {k} ;
    if (ischar (arg))
        arg = lower (arg) ;
        firstchar = min (firstchar, k) ;
        switch arg
            case { 'uniform', 'normal' }
                dist = arg ;
            case { 'range' }
                range = varargin {k+1} ;
                if (isobject (range))
                    range = range.opaque ;
                end
                [rm, rn, type] = gbsize (range) ;
                if (rm*rn > 2)
                    error ('GrB:error', 'range can contain at most 2 entries') ;
                end
                range = gbfull (range, type, 0, struct ('kind', 'full')) ;
            case { 'unsymmetric', 'symmetric', 'hermitian' }
                sym_option = arg ;
            otherwise
                error ('GrB:error', 'unknown option') ;
        end
    end
end

symmetric = isequal (sym_option, 'symmetric') ;
hermitian = isequal (sym_option, 'hermitian') ;
desc.base = 'zero-based' ;

%---------------------------------------------------------------------------
% construct the pattern
%---------------------------------------------------------------------------

if (firstchar == 2)

    % C = GrB.random (A, ...) ;
    A = varargin {1} ;
    if (isobject (A))
        A = A.opaque ;
    end
    [m, n] = gbsize (A) ;
    if ((symmetric || hermitian) && (m ~= n))
        error ('GrB:error', 'input matrix must be square') ;
    end
    [I, J] = gbextracttuples (A, desc) ;
    e = length (I) ;

elseif (firstchar == (4 - (symmetric || hermitian)))

    % C = GrB.random (m, n, d, ...)
    % C = GrB.random (n, d, ... 'symmetric')
    % C = GrB.random (n, d, ... 'hermitian')
    m = gb_get_scalar (varargin {1}) ;
    if (symmetric || hermitian)
        n = m ;
        d = gb_get_scalar (varargin {2}) ;
    else
        n = gb_get_scalar (varargin {2}) ;
        d = gb_get_scalar (varargin {3}) ;
    end
    if (isinf (d))
        % construct a full random matrix
        e = m * n ;
        I = repmat ((int64 (0) : int64 (m-1)), 1, n) ;
        J = repmat ((int64 (0) : int64 (n-1)), m, 1) ;
    else
        % construct a sparse random matrix with about e entries
        e = round (m * n * d) ;
        I = int64 (floor (rand (e, 1) * m)) ;
        J = int64 (floor (rand (e, 1) * n)) ;
    end

else

    error ('GrB:error', 'invalid usage') ;

end

%---------------------------------------------------------------------------
% construct the values
%---------------------------------------------------------------------------

if (isequal (type, 'logical'))

    % X is logical: just pass a single logical 'true' to GrB.build
    X = true ;

else

    % construct the initial random values
    if (isequal (dist, 'uniform'))
        X = rand (e, 1) ;
    else
        X = randn (e, 1) ;
    end

    % scale the values and typecast if requested
    if (~isempty (range))
        lo = double (min (range)) ;
        hi = double (max (range)) ;
        if (gb_contains (type, 'int'))
            % X is signed or unsigned integer
            X = cast (floor ((hi - lo + 1) * X + lo), type) ;
        elseif (~gb_contains (type, 'complex'))
            % X is single or double real
            X = cast ((hi - lo) * X + lo, type) ;
        else
            % X is complex: construct random imaginary values
            if (isequal (dist, 'uniform'))
                Y = rand (e, 1) ;
            else
                Y = randn (e, 1) ;
            end
            X = (hi - lo) * X + lo ;
            Y = (hi - lo) * Y + lo ;
            if (isequal (type, 'single complex'))
                % X is single complex
                X = single (X) ;
                Y = single (Y) ;
            end
            X = complex (X, Y) ;
        end
    end

end

%---------------------------------------------------------------------------
% build the matrix
%---------------------------------------------------------------------------

C = gbbuild (I, J, X, m, n, '2nd', desc) ;

% make it symmetric or hermitian, if requested
L = gbselect ('tril', C, -1) ;
if (symmetric)
    % C = tril (C) + tril (C,-1)'
    C = gbeadd (gbselect ('tril', C, 0), '+', gbtrans (L)) ;
elseif (hermitian)
    % C = L + L' + real (diag (C))
    LT = gbtrans (L) ;
    if (gb_contains (gbtype (LT), 'complex'))
        LT = gbapply ('conj', LT) ;
    end
    D = gbselect ('diag', C, 0) ;
    if (gb_contains (gbtype (D), 'complex'))
        D = gbapply ('creal', D) ;
    end
    C = gbeadd (L, '+', gbeadd (LT, '+', D)) ;
end

