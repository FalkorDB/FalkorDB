function gbtest81 (ghb)
%GBTEST81 test complex operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

fprintf ('gbtest81: test complex operators\n') ;
have_octave = gb_octave ;

% NOTE: min and max for complex matrices are not supported in GraphBLAS

A = sparse (rand (2) + 1i * rand (2))  ;

C = gtb (ghb, A)    %#ok<NOPRT,NASGU>

B = sparse (rand (2) + 1i * rand (2))  ;

A = full (A) ;
B = full (B) ;

C1 = gtb (ghb, A) + gtb (ghb, B) ;
C2 = A+B ;

assert (isequal (C1,C2)) ;

E = rand (2) ;
F = rand (2) ;

C1 = complex (gtb (ghb, E), gtb (ghb, F)) ;
C2 = complex (E,F) ;
assert (isequal (C1,C2)) ;

[complex_binary, complex_unary] = gbtest_complex ;

A (2,1) = B (2,1) ;

% create some complex test matrices

for m = [1 5 10 ]
    for n = [ 1 5 10 ]

        for akind = 1:5
            fprintf ('.') ;
            switch (akind)
                case 1
                    A = complex (zeros (m,n), 0) ;
                case 2
                    A = complex (ones (m,n), 0) ;
                case 3
                    A = complex (-ones (m,n), ones(m,n)) ;
                case 4
                    x = full (sprand(m,n,0.3)) ;
                    y = full (sprand(m,n,0.3)) ;
                    A = complex (x,y) ;
                case 5
                    A = complex (rand (m,n), rand (m,n)) ;
            end

            % test unary ops with complex x
            for k = 1:length (complex_unary)
                op = complex_unary {k} ;
                C1 = gtb_apply (ghb, op, A) ;
                % try built-in methods
                switch (op)
                    case { 'minv' }
                        C2 = 1./A ;
                    case { 'one' }
                        C2 = complex (ones (m, n), 0) ;
                    otherwise
                        C2 = feval (op, A) ;
                end
                err = gbtest_err (C1, C2) ;
                if (~isreal (A) && ...
                    (isequal (op, 'expm1') || isequal (op, 'log1p')))
                    % log1p and expm1 are not accurate in GraphBLAS
                    % for the complex case
                    if (~have_octave)
                        % octave fails here, unsure why
                        assert (err < 1e-5)
                    end
                else
                    assert (err < 1e-13)
                end
            end

            for bkind = 1:6
                switch (bkind)
                    case 1
                        B = complex (zeros (m,n), 0) ;
                    case 2
                        B = complex (ones (m,n), 0) ;
                    case 3
                        B = complex (-ones (m,n), ones(m,n)) ;
                    case 4
                        x = full (sprand(m,n,0.3)) ;
                        y = full (sprand(m,n,0.3)) ;
                        B = complex (x,y) ;
                    case 5
                        B = complex (rand (m,n), rand (m,n)) ;
                    case 6
                        B = gtb (ghb, A) ;
                end

                % test all but the last one, 'cmplex', which requires
                % x,y real
                for k = 1:length (complex_binary)
                    op = complex_binary {k} ;
                    if (isequal (op, 'cmplx'))
                        continue
                    end
                    C1 = gtb_emult (ghb, op, A, B) ;
                    % try built-in methods
                    switch (op)
                        case { '1st' }
                            C2 = A ;
                        case { '2nd', 'any' }
                            C2 = B ;
                        case { 'pair', 'oneb' }
                            C2 = complex (ones (m, n), 0) ;
                        case { '+' }
                            C2 = A+B ;
                        case { '-' }
                            C2 = A-B ;
                        case { 'rminus' }
                            C2 = B-A ;
                        case { '*' }
                            C2 = A .* B ;
                        case { '/' }
                            C2 = A ./ B ;
                        case { '\' }
                            C2 = A .\ B ;
                        case { 'iseq' }
                            C2 = complex (double (A == B), 0) ;
                        case { 'isne' }
                            C2 = complex (double (A ~= B), 0) ;
                        case { '==' }
                            C2 = A == B ;
                        case { '~=' }
                            C2 = A ~= B ;
                        case { 'pow' }
                            C2 = A .^ B ;
                        otherwise
                            error ('unknown') ;
                    end
                    if (have_octave && isequal (op, 'pow'))
                        % skip the error check for octave; it has
                        % different cases for NaNs
                    else
                        assert (gbtest_err (C1, C2) < 1e-12)
                    end
                end

                % test complex(A,B)
                C1 = gtb_emult (ghb, 'cmplx', real (A), real (B)) ;
                C2 = complex (real (A), real (B)) ;
                % octave can return C2 as real, not complex,
                % so just asset this instead:
                assert (norm (abs (C1-C2), 1) == 0)

            end
        end
    end
end

fprintf ('\ngbtest81 (%d): all tests passed\n', ghb) ;

