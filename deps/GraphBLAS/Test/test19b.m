function test19b(fulltest)
%TEST19B test GrB_assign and GrB_*_setElement with many pending operations

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\ntest19b: GrB_assign and setElement, many pending computations\n') ;

debug_status = stat ;
if (debug_status)
    % turn off malloc debugging for this test
    debug_off
end

if (nargin < 1)
    fulltest = 0 ;
end
if (fulltest)
    nt = 3000 ;
else
    nt = 2000 ;
end

for problem = 0:2

    clear Work Work2
    switch (problem)
        case 0
            Corig = sparse (5,5) ;
            d = 1 ;
            ntrials = 100 ;
        case 1
            Corig = sprandn (10,20,0.1) ;
            d = 0.3 ;
            ntrials = nt ;
        case 2
            Corig = sparse (rand (10, 20)) ;
            d = 0.3 ;
            ntrials = nt ;
    end

    rng ('default') ;
    [m n] = size (Corig) ;
    fprintf ('problem %d: C is %d-by-%d, # assign/setElement to do: %d\n', ...
        problem, m, n, ntrials) ;

    if (problem > 0)
        fprintf ('... please wait\n') ;
    end

    for k = 1:ntrials

        c = randi (10,1) ;
        if (c == 10)
            d = struct ('outp', 'replace') ;
        elseif (c == 9)
            d = struct ('outp', 'replace', 'mask', 'complement') ;
        elseif (c == 8)
            d = struct ('mask', 'complement') ;
        else
            d = [ ] ;
        end

        c = randi (3,1) ;
        switch (c)
            case 1
                accum = '' ;
            case 2
                accum = 'plus' ;
            case 3
                accum = 'second' ;
        end

        kind = 0 ;
        c = randi (12,1) ;
        if (c < 8)
            ni = randi (3,1) ;
            nj = randi (3,1) ;
            J = randperm (n, nj) ;
            I = randperm (m, ni) ;
            A = sprand (ni, nj, 0.3) ;
            scalar = 0 ;
        elseif (c == 8)
            % scalar expansion
            ni = 2 ;
            nj = 2 ;
            J = randperm (n, nj) ;
            I = randperm (m, ni) ;
            A = sparse (rand (1)) ;
            scalar = 1 ;
        elseif (c == 9)
            ni = 1 ;
            nj = 1 ;
            I = randperm (m,1) ;
            J = randperm (n,1) ;
            A = sparse (rand (1)) ;
            scalar = 0 ;
        elseif (c == 10)
            ni = 2 ;
            nj = 2 ;
            I = randperm (m,2) ;
            J = randperm (n,2) ;
            A = sparse (rand (2)) ;
            scalar = 0 ;
        elseif (c == 11)
            % column assign
            ni = randi (3,1) ;
            nj = 1 ;
            kind = 1 ;
            J = randperm (n, 1) ;
            I = randperm (m, ni) ;
            A = sprand (ni, 1, 0.3) ;
            scalar = 0 ;
        else % (c == 12)
            % row assign: A is a single *column* vector
            ni = 1 ;
            nj = randi (3,1) ;
            kind = 2 ;
            J = randperm (n, nj) ;
            I = randperm (m, 1) ;
            A = sprand (nj, 1, 0.3) ;
            scalar = 0 ;
        end

        c = randi (2,1) ;
        switch (c)
            case 1
                % no mask
                Mask = [ ] ;
            case 2
                density = rand (1) ;
                if (density < 0.8)
                    if (kind == 0)
                        % mask same size as C: Matrix or Vector assign
                        Mask = (sprand (m, n, 0.3) ~= 0) ;
                    elseif (kind == 1)
                        % mask same size as C(:,j): column assign
                        Mask = (sprand (m, 1, 0.3) ~= 0) ;
                    else % (kind == 2)
                        % mask same size as C(i,:)': row assign
                        Mask = (sprand (n, 1, 0.3) ~= 0) ;
                    end
                else
                    if (kind == 0)
                        % mask same size as C: Matrix or Vector assign
                        Mask = sparse (true (m, n)) ;
                    elseif (kind == 1)
                        % mask same size as C(:,j): column assign
                        Mask = sparse (true (m, 1)) ;
                    else % (kind == 2)
                        % mask same size as C(i,:)': row assign
                        Mask = sparse (true (n, 1)) ;
                    end

                end
        end

        Work (k).A = A ;
        Work (k).I = I ;
        Work (k).J = J ;
        Work (k).Mask = Mask ;
        Work (k).accum = accum ;
        Work (k).desc = d ;
        Work (k).scalar = scalar ;
        Work (k).kind = kind ;

        Work2 (k).A = A ;
        Work2 (k).I = uint64 (I-1) ;
        Work2 (k).J = uint64 (J-1) ;
        Work2 (k).Mask = Mask ;
        Work2 (k).accum = accum ;
        Work2 (k).desc = d ;
        Work2 (k).kind = kind ;

    end

    C3 = Corig ;

    for k = 1:ntrials
        J = Work (k).J ;
        I = Work (k).I ;
        A = Work (k).A ;
        M = Work (k).Mask ;
        accum = Work (k).accum ;
        d = Work (k).desc ;
        scalar = Work (k).scalar ;
        kind = Work (k).kind ;
        if (kind == 0)
            % matrix/vector assign
            C3 = GB_spec_assign (C3, M, accum, A, I, J, d, scalar) ;
        elseif (kind == 1)
            % col assign
            C3 = GB_spec_Col_assign (C3, M, accum, A, I, J, d) ;
        else % (kind == 2)
            % row assign
            C3 = GB_spec_Row_assign (C3, M, accum, A, I, J, d) ;
        end
    end

    % default sparsity
    C2 = GB_mex_assign (Corig, Work2) ; % WORK_ASSIGN
    GB_spec_compare (C2, C3) ;

    % with sparsity control
    for s = 0:15
        C2 = GB_mex_assign (Corig, Work2, [s s]) ;  % WORK_ASSIGN
        GB_spec_compare (C2, C3) ;
    end

    % default sparsity but both C and M sparse
    C2 = GB_mex_assign (Corig, Work2, [2 2]) ;  % WORK_ASSIGN
    GB_spec_compare (C2, C3) ;

    % default sparsity but C sparse and M bitmap/full
    C2 = GB_mex_assign (Corig, Work2, [2 8]) ;  % WORK_ASSIGN
    GB_spec_compare (C2, C3) ;

end

if (debug_status)
    debug_on
end
fprintf ('\ntest19b: all tests passed\n') ;

