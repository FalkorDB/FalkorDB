% Run the DNN benchmarks

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% reset to the default number of threads
clear all
GrB.clear ;

rootdir = '/raid/hyper/GraphChallenge/dnn_data/stuff' ;
ncores = demo_nproc ;

fprintf ('# of cores :  %d\n', ncores) ;
GrB.format ('by row') ;

for id = 1:12
    % load the problem
    tic ;
    fname = sprintf ('%s/HPEC19_dnn_%02d.mat', rootdir, id) ;
    fprintf ('\nProblem: %s\n', fname) ;
    load (fname) ;
    disp (Problem)
    W = Problem.layers ;
    bias = Problem.bias ;
    Y0 = Problem.featureVectors ;
    trueCat = Problem.trueCategories ;
    nlayers = Problem.nlayers ;
    clear Problem
    DNNedges = 0 ;
    for k = 1:nlayers
        DNNedges = DNNedges + GrB.entries (W {k}) ;
        bias {k} = full (bias {k}) ;
    end
    tload = toc ;
    fprintf ('load time:          %14.2f sec\n', tload) ;

    % convert to GraphBLAS
    tic ;
    [W, bias, Y0] = dnn_builtin2gb (W, bias, Y0) ;
    tsetup = toc ;
    fprintf ('total convert time: %14.2f sec\n', tsetup) ;
    t1 = 0 ;

    for nthreads = [1 2 4 8 16 20 32 40 64 128]

        if (nthreads > 2*ncores)
            break ;
        end
        GrB.threads (nthreads) ;

        % solve the DNN
        tic
        Y = GrB.dnn (W, bias, Y0) ;
        challengeRunTime = toc ;
        if (nthreads == 1)
            t1 = challengeRunTime ;
        end

        % check the results
        NfeatureVectors = size (Y0, 1) ;
        challengeRunRate = NfeatureVectors*DNNedges/challengeRunTime;
        % fprintf ('total run time:     %14.2f sec\n', challengeRunTime) ;
        % fprintf ('run rate:           %14.2f billion\n', challengeRunRate / 1e9) ;
        fprintf ('threads: %3d time: %10.2f sec speedup: %8.2f rate: %10.2f billion\n', ...
            nthreads, challengeRunTime, t1 / challengeRunTime, challengeRunRate / 1e9) ;
        [categories, ~, ~] = find (sum (Y,2)) ;
        assert (isequal (categories, trueCat)) ;
        clear Y
        clear categories

    end
    fprintf ('problem %d OK\n', id) ;

    clear W Y Y0
end

% restore the default # of threads
GrB.clear ;

