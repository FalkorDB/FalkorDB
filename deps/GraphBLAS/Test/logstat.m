function logstat (testscript, threads, jit_controls, factory_controls, pji_controls)
%LOGSTAT run a GraphBLAS test and log the results to log.txt 
%
% logstat (testscript, threads, jit_controls, factory_controls, pji_controls)
%
% threads: defaults to threads{1} = [4 1], which uses 4 threads and a tiny
% chunk size of 1.
%
% jit_controls: a parameter for GB_mex_jit_control (0 to 5: off, pause, run,
% load, on).  JIT kernels from the prior test are always cleared from the JIT
% hash table, and then the JIT is renabled.  This is to prevent a sequence of
% many tests to run out of memory from loading too many JIT kernels.  If
% jit_controls is empty, the JIT control is left unchanged.
%
% factory_controls: 1 to enable the factory kernels, 0 to disable them.
% If empty, default is enabled.
%
% pji_controls: a list of integers in the range 0 to 7, where each integer
% is a 3-bit number with [pji]_control.  Defaults to [0]

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

GB_mex_finalize ;
GB_mex_factory_control (1) ;
[debug, compact, malloc, covered] = GB_mex_debug ;

test_coverage = (~isempty (strfind (pwd, 'Tcov'))) ;
if (test_coverage)
    global GraphBLAS_debug GraphBLAS_grbcov
%   global GraphBLAS_grbcovs GraphBLAS_scripts GraphBLAS_times
end

% default JIT controls
if (nargin < 3)
    jit_controls = [ ] ;
end
if (isempty (jit_controls))
    jit_controls {1} = 4 ;      % JIT on
%   jit_controls {2} = 0 ;      % JIT off
%   jit_controls {3} = 4 ;      % JIT on
end

% default factory controls
if (nargin < 4)
    factory_controls = [ ] ;
end
if (isempty (factory_controls))
    factory_controls {1} = 1 ;  % factory on
%   factory_controls {2} = 1 ;  % factory on
%   factory_controls {3} = 0 ;  % factory off
end

% default pji_controls
if (nargin < 5)
    pji_controls = [ 0 ] ;
end

if (0)
    % enable this to run with all pji_controls
    pji_controls = 0:7 ;
end

if (0)
    % enable this to run with all JIT / factory controls
    jall = {4,3,2,1,0,4,3,2,1,0} ;
    fall = {1,1,1,1,1,0,0,0,0,0} ;
    jit_controls    = jall ;
    factory_controls = fall ;
end

if (nargin < 2)
    % by default, use 4 threads and a tiny chunk size of 1
    threads = [ ] ;
end

if (isempty (threads))
    threads {1} = [4 1] ;
else
    % only the # of threads is specified; also set the chunk size to 1
    if (isscalar (threads) && isnumeric (threads))
        threads = max (threads, 1) ;
        t {1} = [threads 1] ;
        threads = t ;
    end
end

try
    n = grblines ;  % total # of lines in the test coverage
catch
    n = 0 ;
end
if (nargin == 0)
    fprintf (   'total blocks: %d\n', n) ;
    f = fopen ('log.txt', 'a') ;
    fprintf (f, 'total blocks: %d\n', n) ;
    fclose (f) ;
end

% f = fopen ('log.txt', 'a') ;
% fprintf (f, '\n') ;

for pji_control_trials = 1:length(pji_controls)

    clear ctrl
    pji_control = pji_controls (pji_control_trials) ;
    p_control = 32 * (bitand (pji_control, 4) ~= 0) + 32 ;
    j_control = 32 * (bitand (pji_control, 2) ~= 0) + 32 ;
    i_control = 32 * (bitand (pji_control, 1) ~= 0) + 32 ;
    ctrl.p_control = p_control ;
    ctrl.j_control = j_control ;
    ctrl.i_control = i_control ;
    ctrl = GB_mex_control (ctrl) ;

    for control_trial = 1:length (jit_controls)
        for trial = 1:length (threads)

            GB_mex_finalize ;
            jit_control = jit_controls {control_trial} ;
            factory_control = factory_controls {control_trial} ;
            if (~isempty (jit_control))
                GB_mex_jit_control (jit_control) ;
            end
            if (isempty (factory_control))
                factory_control = 1 ;
            end
            GB_mex_factory_control (factory_control) ;
            fprintf ('\nTrial: jit: %d factory: %d pji: %d:(%d,%d,%d)\n', ...
                jit_control, factory_control, pji_control, ...
                p_control, j_control, i_control) ;

            clast = grb_get_coverage ;

            nthreads_and_chunk = threads {trial} ;
            nthreads = nthreads_and_chunk (1) ;
            chunk    = nthreads_and_chunk (2) ;
            nthreads_set (nthreads, chunk) ;

            if (nargin == 0)
                f = fopen ('log.txt', 'a') ;
                fprintf (f, '\n----------------------------------------------');
                if (debug)
                    fprintf (f, ' [debug]') ;
                end
                if (compact)
                    fprintf (f, ' [compact]') ;
                end
                if (malloc)
                    fprintf (f, ' [malloc]') ;
                end
                if (covered)
                    fprintf (f, ' [cover]') ;
                end
                fprintf (f, '\n') ;
                fclose (f) ;
                return
            end

            fprintf ('\n======== test: %-10s ', testscript) ;

            if (debug)
                fprintf (' [debug]') ;
            end
            if (compact)
                fprintf (' [compact]') ;
            end
            if (malloc)
                fprintf (' [malloc]') ;
            end
            if (covered)
                fprintf (' [cover]') ;
            end
            fprintf (' [nthreads: %d chunk: %g]', nthreads, chunk) ;
            fprintf (' jit: %d', GB_mex_jit_control) ;
            fprintf (' factory: %d\n', GB_mex_factory_control) ;

            t1 = tic ;
            runtest (testscript)
            t = toc (t1) ;

            f = fopen ('log.txt', 'a') ;

            s = datestr (now) ;

            % trim the year from the date
            s = s ([1:6 12:end]) ;

            fprintf (   '%s %-11s %d:(%d,%d,%d) %7.1f sec', s, testscript, ...
                pji_control, p_control, j_control, i_control, t) ;
            fprintf (f, '%s %-11s %d:(%d,%d,%d) %7.1f sec', s, testscript, ...
                pji_control, p_control, j_control, i_control, t) ;

            if (test_coverage)

                if (isempty (GraphBLAS_debug))
                    GraphBLAS_debug = false ;
                end
                if (~isempty (GraphBLAS_grbcov))
                    c = sum (GraphBLAS_grbcov > 0) ;
                    if (c == n)
                        % full coverage reached with this test
                        fprintf (   '%5d:   all  100%% %7.1f/s', ...
                            c - clast, (c-clast) / t) ;
                        fprintf (f, '%5d:   all  100%% %7.1f/s', ...
                            c - clast, (c-clast) / t) ;
                    elseif (c == clast)
                        % no new coverage at all with this test
                        fprintf (   '     : %5d %4.1f%%', n-c, 100 * (c/n)) ;
                        fprintf (f, '     : %5d %4.1f%%', n-c, 100 * (c/n)) ;
                    else
                        crel = 100 * (c/n) ;
                        if (crel < 100 && crel > 99.9)
                            crel = 99.9 ;
                        end
                        fprintf (   '%5d: %5d %4.1f%% %7.1f/s', ...
                            c - clast, n-c, crel, (c-clast) / t) ;
                        fprintf (f, '%5d: %5d %4.1f%% %7.1f/s', ...
                            c - clast, n-c, crel, (c-clast) / t) ;
                    end
                    if (debug)
                        fprintf (' [debug]') ;
                    end
                    if (compact)
                        fprintf (' [compact]') ;
                    end
                    if (malloc)
                        fprintf (' [malloc]') ;
                    end
                    if (covered)
                        fprintf (' [cover]') ;
                    end
                end
            end

            fprintf (   '\n') ;
            fprintf (f, '\n') ;
            fclose (f) ;
        end
    end
end

