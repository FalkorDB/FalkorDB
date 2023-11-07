function test51
%TEST51 test GxB_subassign, multiply operations

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\n-----------performance test GB_mex_subassign, multiple ops\n') ;

[save save_chunk] = nthreads_get ;
chunk = 4096 ;
nthreads = feature_numcores ;
nthreads_set (nthreads, chunk) ;

rng ('default')

for problem = 1:5

    fprintf ('\n----------------------------------------------------- %d\n',...
    problem) ;

    switch (problem)
    case 1
        Corig = sprand (1000, 1, 0.1) ;
        nwork = 10 ;
        m_max = 1000 ;
        n_max = 1 ;
        d = 0.1 ;
    case 2
        Prob = ssget ('HB/west0067') ; Corig = Prob.A ;
        nwork = 1000 ;
        m_max = 64 ;
        n_max = 64 ;
        d = 0.1 ;
    case 3
        Corig = abs (sprand (80, 80, 0.4)) ;
        nwork = 1000 ;
        m_max = 64 ;
        n_max = 64 ;
        d = 0.1 ;
    case 4
        Corig = sparse (rand (50000, 500)) ;
        nwork = 50 ;
        % [m n] = size (A) ;
        [m n] = size (Corig) ;
        m_max = m ;
        n_max = n ;
        d = 100 / (m*n) ;
    case 5
        Prob = ssget (2662)
        Corig = Prob.A ;
        nwork = 100 ;
        m_max = 1024 ;
        n_max = 1024 ;
        d = 10000 / (ni*nj) ;
    end

    Corig = abs (Corig) ;
    [m n] = size (Corig) ; 

    for opkind = 1 % 1:3

        fprintf ('\n--------------------------\n') ;
        switch opkind
        case 1
            fprintf ('C(I,J) = accum(C(I,J),A) one op, assemble at end\n') ;
        case 2
            fprintf ('C(I,J) = accum(C(I,J),A) two ops, change every 10th\n') ;
        case 3
            fprintf ('C(I,J) = A no accum\n') ;
        end

        nz = 0 ;
        ni_min = inf ;
        ni_max = -inf ;
        nj_min = inf ;
        nj_max = -inf ;
        nz_min = inf ;
        nz_max = -inf ;

        clear Work

        for k = 1:nwork
            ni = double (irand (1, min (m,m_max))) ;
            nj = double (irand (1, min (n,n_max))) ;
            ni_min = min (ni_min, ni) ;
            nj_min = min (nj_min, nj) ;
            ni_max = max (ni_max, ni) ;
            nj_max = max (nj_max, nj) ;

            if (d == 1)
                A = sparse (rand (ni, nj)) ;
            else
                A = sprand (ni, nj, d) ;
            end
            nz_min = min (nz_min, nnz (A)) ;
            nz_max = max (nz_max, nnz (A)) ;
            if (opkind == 1)
                op = 'plus' ;
            else
                if (mod (k,5) == 0)
                    op = 'max' ;
                else
                    op = 'plus' ;
                end
            end
            nz = nz + nnz (A) ;
            I = randperm (m,ni) ;
            J = randperm (n,nj) ;
            Work (k).A = A ;
            Work (k).I = I ;
            Work (k).J = J ;
            Work (k).accum = op ;
        end
        fprintf ('number of C(I,J) = ... to do: %d\n', nwork) ;
        fprintf ('ni: [ %d to %d]\n', ni_min, ni_max) ;
        fprintf ('nj: [ %d to %d]\n', nj_min, nj_max) ;
        fprintf ('nz: [ %d to %d]\n', nz_min, nz_max) ;
        fprintf ('C is %d-by-%d nnz(A): %d  nz to add: %d  matrices: %d\n', ...
            m, n, nnz (Corig), nz, nwork) ;

        Work2 = Work ;
        for k = 1:nwork
            Work2 (k).I = uint64 (Work2 (k).I - 1) ;
            Work2 (k).J = uint64 (Work2 (k).J - 1) ;
        end
        tic
        C2 = GB_mex_subassign (Corig, Work2) ;  % WORK_ASSIGN
        t1 = toc ;
        fprintf ('GraphBLAS time: %g\n', t1) ;
        fprintf ('final nnz: %d\n', nnz (C2.matrix)) ;

        fprintf ('start built-in methods...\n') ;
        tic
        C = Corig ;
        % full (C)
        for k = 1:nwork
            I = Work (k).I ;
            J = Work (k).J ;
            A = Work (k).A ;
            % Afull = full (A)
            if (isequal (Work (k).accum, 'plus'))
                % fprintf (' %d : plus\n', k) ;
                C (I,J) = C (I,J) + A ;
            elseif (isequal (Work (k).accum, 'max'))
                % fprintf (' %d : max\n', k) ;
                C (I,J) = max (C (I,J),A) ;
            else
                C (I,J) = A ;
            end
            % full (C)
        end
        t2 = toc ;
        fprintf ('built-in methods time: %g\n', t2) ;
        fprintf ('GraphBLAS speedup: %g\n', t2/t1) ;

        % C2.matrix
        % C - C2.matrix

        % C2 = full (C2.matrix)
        % C-C2
        % assert (isequal (C, C2)) ;
        assert (isequal (C, C2.matrix)) ;
    end

end

fprintf ('\ntest51: all tests passed\n') ;

nthreads_set (save, save_chunk) ;
