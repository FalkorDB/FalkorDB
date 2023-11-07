function gbtest102
%GBTEST102 test horzcat, vertcat, cat, cell2mat, mat2cell, num2cell

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

have_octave = gb_octave ;

rng ('default') ;
A = GrB (rand (2)) ;
B = GrB (speye (2)) ;
C1 = [A B] ;
C2 = [double(A) double(B)] ;
assert (isequal (C1, C2)) ;

C1 = [A ; B]  ;
C2 = [double(A) ; double(B)] ;
assert (isequal (C1, C2)) ;

if (~have_octave)
    % num2cell works differently in octave
    S1 = num2cell (C1) ;
    S2 = num2cell (C2) ;
    assert (isequal (S1, S2)) ;

    S1 = num2cell (C1, 1) ;
    S2 = num2cell (C2, 1) ;
    assert (isequal (S1, S2)) ;

    S1 = num2cell (C1, 2) ;
    S2 = num2cell (C2, 2) ;
    assert (isequal (S1, S2)) ;

    S1 = num2cell (C1, [1 2]) ;
    S2 = num2cell (C2, [1 2]) ;
    assert (isequal (S1, S2)) ;

    S1 = num2cell (C1, [2 1]) ;
    S2 = num2cell (C2, [2 1]) ;
    assert (isequal (S1, S2)) ;
end

for n = 100:100:1000
    fprintf ('.') ;
    for d = [1e-4 0.01]

        % create random GrB matrices
        A1 = GrB.random (n, n, d) ;
        A2 = GrB.random (n, n, d) ;
        A3 = GrB.random (n, n, d) ;
        A4 = GrB.random (n, n, d) ;

        % convert to built-in double sparse matrices
        B1 = double (A1) ;
        B2 = double (A2) ;
        B3 = double (A3) ;
        B4 = double (A4) ;

        C1 = [A1 A2 ; A3 A4] ;  % using GrB horzcat and vertcat
        C2 = [B1 B2 ; B3 B4] ;  % using built-in horzcat and vercat
        assert (isequal (C1, C2)) ;

        % test mat2cell
        S1 = mat2cell (C1, [n n], [n n]) ;
        S2 = mat2cell (C2, [n n], [n n]) ;
        assert (isequal (S1, S2)) ;

        % test GrB.cell2mat
        S1 = cell (2,2) ;
        S1 {1,1} = A1 ;
        S1 {1,2} = A2 ;
        S1 {2,1} = A3 ;
        S1 {2,2} = A4 ;
        C3 = GrB.cell2mat (S1) ;

        % and compare with the built-in cell2mat
        S2 = cell (2,2) ;
        S2 {1,1} = B1 ;
        S2 {1,2} = B2 ;
        S2 {2,1} = B3 ;
        S2 {2,2} = B4 ;
        C4 = cell2mat (S2) ;
        assert (isequal (C3, C2)) ;
        assert (isequal (C3, C4)) ;

        % test cat
        C1 = [A1 A2 A3 A4] ;            % GrB/horzcat
        C2 = cat (2, A1, A2, A3, A4) ;  % GrB/cat
        C3 = [B1 B2 B3 B4] ;            % built-in/horzcat
        C4 = cat (2, B1, B2, B3, B4) ;  % built-in/cat
        assert (isequal (C1, C2)) ;
        assert (isequal (C1, C3)) ;
        assert (isequal (C1, C4)) ;

        C1 = [A1 ; A2 ; A3 ; A4] ;      % GrB/vertcat
        C2 = cat (1, A1, A2, A3, A4) ;  % GrB/cat
        C3 = [B1 ; B2 ; B3 ; B4] ;      % built-in/vertcat
        C4 = cat (1, B1, B2, B3, B4) ;  % built-in/cat
        assert (isequal (C1, C2)) ;
        assert (isequal (C1, C3)) ;
        assert (isequal (C1, C4)) ;
    end
end

for n1 = [10 100 1000]
    for n2 = [20 100 1000]
        for n3 = [1 5 100]
            fprintf ('.') ;
            for m1 = [10 100 1000]
                for m2 = [20 100 1000]
                    for d = [1e-4 0.01 0.5 inf]

                        if (d > 0.1)
                            if (max ([n1 n2 n3 m1 m2]) > 100)
                                continue ;
                            end
                        end

                        S {1,1} = GrB.random (m1, n1, d) ;
                        S {1,2} = GrB.random (m1, n2, d) ;
                        S {1,3} = GrB.random (m1, n3, d) ;
                        S {2,1} = GrB.random (m2, n1, d) ;
                        S {2,2} = GrB.random (m2, n2, d) ;
                        S {2,3} = GrB.random (m2, n3, d) ;

                        T {1,1} = double (S {1,1}) ;
                        T {1,2} = double (S {1,2}) ;
                        T {1,3} = double (S {1,3}) ;
                        T {2,1} = double (S {2,1}) ;
                        T {2,2} = double (S {2,2}) ;
                        T {2,3} = double (S {2,3}) ;

                        C1 = GrB.cell2mat (S) ;
                        C2 = cell2mat (T) ;
                        assert (isequal (C1, C2)) ;

                        S1 = mat2cell (C1, [m1 m2], [n1 n2 n3]) ;
                        S2 = mat2cell (C2, [m1 m2], [n1 n2 n3]) ;
                        assert (isequal (S1, S2)) ;
                    end
                end
            end
        end
    end
end

fprintf ('\ngbtest102: all tests passed\n') ;
