function test53(fulltests)
%TEST53 test GrB_Matrix_extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 1)
    fulltests = 0 ;
end

if (fulltests)
    fprintf ('\n==== test53: exhaustive test for GrB_Matrix_extract:\n') ;
else
    fprintf ('\n==== test53: quick test for GrB_Matrix_extract:\n') ;
end

[binops, ~, ~, types, ~, ~] = GB_spec_opsall ;
accum_ops = binops.all ;
types = types.all ;

problems = [
    10,    1,    7,  -5, 100
    10,    8,   40,  -5, 100
    10,  20,  100,  -99, 200
    100, 200, 1000, -99, 200
     50,  50,  500,  -2, 3
    ] ;

% try several problems
for k0 = 1:size (problems,1) ;

    % create nnz triplets for a matrix of size nrows-by-nrows
    nrows = problems (k0,1) ;
    ncols = problems (k0,2) ;
    nnz = problems (k0,3) ;
    y1 = problems (k0,4) ;
    y2 = problems (k0,5) ;

    % create A
    rng ('default') ;
    I = irand (0, nrows-1, nnz, 1) ;
    J = irand (0, ncols-1, nnz, 1) ;
    Y = y2 * rand (nnz, 1) + y1 ;
    clear A
    A.matrix = sparse (double (I)+1, double (J)+1, Y, nrows, ncols) ;

    % create Cin; note that it has the same dimensions as A, so if A
    % gets transpose, Cin must also be transposed (so use Cin2 instead)
    I = irand (0, nrows-1, nnz, 1) ;
    J = irand (0, ncols-1, nnz, 1) ;
    Y = y2 * rand (nnz, 1) + y1 ;
    clear Cin
    Cin.matrix = sparse (double (I)+1, double (J)+1, Y, nrows, ncols) ;
    clear I J

    clear Cin2
    Cin2.matrix = Cin.matrix' ;

    clear Cempty
    Cempty.matrix = sparse (nrows, ncols) ;
    Cempty2.matrix = Cempty.matrix' ;

    % create a boolean Mask with roughly the same density as A and Cin
    Mask = cast (sprandn (nrows, ncols, nnz/(nrows*ncols)), 'logical') ;

    fprintf ('\nnrows: %d ncols %d nnz %d ymin %g ymax %g\n', ...
        nrows, ncols, nnz, min (Y), max (Y)) ;

    if (fulltests)
        k1_list = [1:length(types)] ;
    else
        k1_list = 11 ;
    end

    % try every type for A
    for k1 = k1_list % 1:length(types)
        atype = types {k1} ;
        A.class = atype ;
        Cempty.class = atype ;
        Cempty2.class = atype ;

        % C = A (:,:)
        C = GB_mex_Matrix_extract  (Cempty, [ ], [ ], A, [ ], [ ], [ ]) ;
        assert (GB_spok (C.matrix*1) == 1) ;
        S = GB_spec_Matrix_extract (Cempty, [ ], [ ], A, [ ], [ ], [ ]) ;
        assert (isequal (C.class, A.class)) ;
        assert (isequal (C.class, S.class)) ;
        assert (isequal (full (double (C.matrix)), double (S.matrix))) ;
        if (isequal (C.class, 'double'))
            assert (isequal (C.matrix, A.matrix)) ;
        end

        % C = A (:,:)'
        clear D
        D = struct ('inp0', 'tran') ;
        C = GB_mex_Matrix_extract  (Cempty2, [ ], [ ], A, [ ], [ ], D) ;
        assert (GB_spok (C.matrix*1) == 1) ;
        S = GB_spec_Matrix_extract (Cempty2, [ ], [ ], A, [ ], [ ], D) ;
        assert (isequal (C.class, A.class)) ;
        assert (isequal (C.class, S.class)) ;
        assert (isequal (full (double (C.matrix)), double (S.matrix))) ;
        if (isequal (C.class, 'double'))
            assert (isequal (C.matrix, A.matrix')) ;
        end

        % C<Mask> = A (:,:)
        C = GB_mex_Matrix_extract  (Cempty, Mask, [ ], A, [ ], [ ], [ ]) ;
        assert (GB_spok (C.matrix*1) == 1) ;
        S = GB_spec_Matrix_extract (Cempty, Mask, [ ], A, [ ], [ ], [ ]) ;
        assert (isequal (C.class, A.class)) ;
        assert (isequal (C.class, S.class)) ;
        assert (isequal (full (double (C.matrix)), double (S.matrix))) ;
        if (isequal (C.class, 'double'))
            assert (isequal (C.matrix .* Mask, (A.matrix).*Mask)) ;
        end

        % C<Mask> = A (:,:)'
        clear D
        D = struct ('inp0', 'tran') ;
        C = GB_mex_Matrix_extract  (Cempty2, Mask', [ ], A, [ ], [ ], D) ;
        assert (GB_spok (C.matrix*1) == 1) ;
        S = GB_spec_Matrix_extract (Cempty2, Mask', [ ], A, [ ], [ ], D) ;
        assert (isequal (C.class, A.class)) ;
        assert (isequal (C.class, S.class)) ;
        assert (isequal (full (double (C.matrix)), double (S.matrix))) ;
        if (isequal (C.class, 'double'))
            assert (isequal (C.matrix .* Mask', (A.matrix') .* Mask')) ;
        end

        if (fulltests)
            k2_list = [1:length(types)] ;
        else
            k2_list = unique ([11 irand(2,length(types),1,1)]) ;
        end

        % try every type for Cin
        for k2 = k2_list
            cintype = types {k2} ;
            Cin2.class = cintype ;
            Cin.class = cintype ;

            fprintf ('%s', cintype) ;

            if (fulltests)
                k3_list = 1:length (accum_ops) ;
            else
                k3_list = unique ([1 5 irand(2,length(accum_ops),1,1)]) ;
            end

            % try every operator
            for k3 = k3_list
                op = accum_ops {k3} ;
                fprintf ('.') ;
                if (isequal (op, 'hypot'))
                    continue
                end
                % op

                if (fulltests)
                    k4_list = [1:length(types)] ;
                else
                    k4_list = unique ([11 irand(2,length(types),1,1)]) ;
                end

                % try every operator type
                for k4 = k4_list
                    optype = types {k4} ;

                    clear accum
                    accum.opname = op ;
                    accum.optype = optype ;

                    if (GB_spec_is_positional (accum))
                        continue ;
                    end

                    try
                        GB_spec_operator (accum) ;
                    catch
                        continue
                    end

                    z = GB_mex_cast (1, optype) ;
                    opint = isinteger (z) || islogical (z) ;

                    % try several I's
                    for k5 = 1:4

                        switch (k5)
                            case 1
                                I = [ ] ;
                            case 2
                                I = uint64 (1 + floor(nrows/2)) ;
                                if (I+2 < nrows)
                                    I = [I I+2] ;
                                end
                            case 3
                                I = uint64 (randperm (nrows)) ;
                            case 4
                                I = uint64 (min (4, nrows-1)) ;
                        end
                        II = I ;
                        if (isempty (II))
                            II = 1:nrows ;
                        end
                        ni = length (II) ;

                        if (size (A,2) == 1)
                            k6_cases = 2 ;
                        else
                            k6_cases = 4 ;
                        end

                        % try several J's
                        for k6 = 1:k6_cases

                            switch (k6)
                                case 1
                                    J = [ ] ;
                                case 2
                                    J = uint64 (1 + floor(ncols/2)) ;
                                    if (J+2 < ncols)
                                        J = [J J+2] ;
                                    end
                                case 3
                                    J = uint64 (randperm (ncols)) ;
                                case 4
                                    J = uint64 (1) ;
                            end
                            JJ = J ;
                            if (isempty (JJ))
                                JJ = 1:ncols ;
                            end
                            nj = length (JJ) ;

                            clear Csub Csub2

                            Csub.matrix   = Cin.matrix   (1:ni,1:nj) ;
                            Csub.class    = Cin.class ;

                            Csub2.matrix  = Cin2.matrix  (1:nj,1:ni) ;
                            Csub2.class   = Cin2.class ;

                            for A_is_hyper = 0:1
                            for A_is_csc   = 0:1
                            A.is_hyper = A_is_hyper ;
                            A.is_csc   = A_is_csc   ;

                            % C = op (Csub,A(I,J))
                            C = GB_mex_Matrix_extract  (Csub, [ ], accum, ...
                                A, I-1, J-1, [ ]) ;
                            assert (GB_spok (C.matrix*1) == 1) ;
                            C.matrix = C.matrix * 1 ;
                            S = GB_spec_Matrix_extract (Csub, [ ], accum,  ...
                                A, I, J, [ ]) ;
                            assert (isequal (C.class, cintype)) ;
                            assert (isequal (C.class, S.class)) ;
                            if (~(isequalwithequalnans (...
                                full (double (C.matrix)), ...
                                double (S.matrix))))
                                assert (false)
                            end

                            A_is_vector = (size (A.matrix,2) == 1 && ...
                                isequal (J, 1) && A.is_csc && ~A.is_hyper) ;

                            if (A_is_vector)
                                % A is a column vector; test Vector_extract
                                % C = op (Csub,A(I,1))
                                C = GB_mex_Vector_extract  (Csub, [ ], ...
                                    accum, A, I-1, [ ]) ;
                                assert (GB_spok (C.matrix*1) == 1) ;
                                S = GB_spec_Vector_extract (Csub, [ ], ...
                                    accum, A, I, [ ]) ;
                                assert (isequal (C.class, cintype)) ;
                                assert (isequal (C.class, S.class)) ;
                                assert (isequalwithequalnans (...
                                    full (double (C.matrix)), ...
                                    double (S.matrix))) ;
                            end

                            if (length (J) == 1)
                                % J is a scalar, test Col_extract
                                % C = op (Csub,A(I,j))
                                C = GB_mex_Col_extract  (Csub, [ ], ...
                                    accum, A, I-1, J-1, [ ]) ;
                                assert (GB_spok (C.matrix*1) == 1) ;
                                S = GB_spec_Col_extract (Csub, [ ], ...
                                    accum, A, I, J, [ ]) ;
                                assert (isequal (C.class, cintype)) ;
                                assert (isequal (C.class, S.class)) ;
                                assert (isequalwithequalnans (...
                                    full (double (C.matrix)), ...
                                    double (S.matrix))) ;
                            end

                            % C = op (Csub,A(J,I)')
                            clear D
                            D = struct ('inp0', 'tran') ;

                            C = GB_mex_Matrix_extract  (Csub2, [ ], accum,  ...
                                A, J-1, I-1, D) ;
                            assert (GB_spok (C.matrix*1) == 1) ;
                            S = GB_spec_Matrix_extract (Csub2, [ ], accum,  ...
                                A, J, I, D) ;
                            assert (isequal (C.class, cintype)) ;
                            assert (isequal (C.class, S.class)) ;
                            assert (isequalwithequalnans (...
                                full (double (C.matrix)), ...
                                double (S.matrix))) ;

                            if (length (I) == 1)
                                % I is a scalar, test Col_extract
                                % C = op (Csub,A(i,J)')
                                C = GB_mex_Col_extract  (Csub2, [ ], ...
                                    accum, A, J-1, I-1, D) ;
                                assert (GB_spok (C.matrix*1) == 1) ;
                                S = GB_spec_Col_extract (Csub2, [ ], ...
                                    accum, A, J, I, D) ;
                                assert (isequal (C.class, cintype)) ;
                                assert (isequal (C.class, S.class)) ;
                                assert (isequalwithequalnans (...
                                    full (double (C.matrix)), ...
                                    double (S.matrix))) ;
                            end

                            % try with a Mask (Mask must be sparse; logical and
                            % double)

                            for k7 = [1 11]
                                mask_class = types {k7} ;
                                M = cast (Mask, mask_class) ;
                                Msub  = M (1:ni, 1:nj) ;

                                % C = op (Csub2,A (I,J))
                                C = GB_mex_Matrix_extract  (Csub, Msub,  ...
                                    accum, A, I-1, J-1, [ ]) ;
                                assert (GB_spok (C.matrix*1) == 1) ;
                                S = GB_spec_Matrix_extract (Csub, Msub,  ...
                                    accum, A, I, J, [ ]) ;
                                assert (isequal (C.class, cintype)) ;
                                assert (isequal (C.class, S.class)) ;
                                assert (isequalwithequalnans (...
                                    full (double (C.matrix)), ...
                                    double (S.matrix))) ;

                                if (A_is_vector)
                                    % A is a column vector; test Vector_extract
                                    % C = op (Csub,A(I,1))
                                    C = GB_mex_Vector_extract  (Csub, Msub, ...
                                        accum, A, I-1, [ ]) ;
                                    assert (GB_spok (C.matrix*1) == 1) ;
                                    S = GB_spec_Vector_extract (Csub, Msub, ...
                                        accum, A, I, [ ]) ;
                                    assert (isequal (C.class, cintype)) ;
                                    assert (isequal (C.class, S.class)) ;
                                    assert (isequalwithequalnans (...
                                        full (double (C.matrix)), ...
                                        double (S.matrix))) ;
                                end

                                if (length (J) == 1)
                                    % J is a scalar, test Col_extract
                                    % C = op (Csub,A(I,j))
                                    C = GB_mex_Col_extract  (Csub, Msub, ...
                                        accum, A, I-1, J-1, [ ]) ;
                                    assert (GB_spok (C.matrix*1) == 1) ;
                                    S = GB_spec_Col_extract (Csub, Msub, ...
                                        accum, A, I, J, [ ]) ;
                                    assert (isequal (C.class, cintype)) ;
                                    assert (isequal (C.class, S.class)) ;
                                    assert (isequalwithequalnans (...
                                        full (double (C.matrix)), ...
                                        double (S.matrix))) ;
                                end

                                % C = op (Csub,A(J,I)')
                                clear D
                                D = struct ('inp0', 'tran') ;
                                C = GB_mex_Matrix_extract  (Csub2, Msub',  ...
                                    accum, A, J-1, I-1, D) ;
                                assert (GB_spok (C.matrix*1) == 1) ;
                                S = GB_spec_Matrix_extract (Csub2, Msub',  ...
                                    accum, A, J, I, D) ;
                                assert (isequal (C.class, cintype)) ;
                                assert (isequal (C.class, S.class)) ;
                                assert (isequalwithequalnans (...
                                    full (double (C.matrix)), ...
                                    double (S.matrix))) ;

                                if (length (I) == 1)
                                    % I is a scalar, test Col_extract
                                    % C = op (Csub,A(i,J)')
                                    C = GB_mex_Col_extract  (Csub2, Msub', ...
                                        accum, A, J-1, I-1, D) ;
                                    assert (GB_spok (C.matrix*1) == 1) ;
                                    S = GB_spec_Col_extract (Csub2, Msub', ...
                                        accum, A, J, I, D) ;
                                    assert (isequal (C.class, cintype)) ;
                                    assert (isequal (C.class, S.class)) ;
                                    assert (isequalwithequalnans (...
                                        full (double (C.matrix)), ...
                                        double (S.matrix))) ;
                                end
                            end

                            end
                            end

                        end
                    end
                end
            end
        end
    end
end

fprintf ('\ntest53: all tests passed\n') ;

