function gbtest132 (ghb)
%GBTEST132 test loading of MAT files from prior versions of GraphBLAS

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% For future updates:  this test creates /tmp/gbtest*.mat files.  One pass of
% gbtest calls this method with ghb = 0, 1, and 2.  When it finishes, move
% the /tmp/gbtest*.mat files into the gbtest132_matfiles folder, and then add
% the latest version to the "versions = { ... }" cell below.

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

% each prior version of GraphBLAS was used to create these matrices and files:
[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;
load (fullfile (filepath, './matrix/west0479_correct.mat')) ;
A = Problem.A ;
Sparse = gtb (ghb, A) ;
S = delsq (numgrid ('B', 100)) ;    % does not appear in octave
n = 2^50 ;
Hyper = gtb (ghb, n,n) ;
m = size (S,1) ;
Hyper (1:m,1:m) = S ;
Bitmap = gtb (ghb, S (1:10, 1:10), 'bitmap') ;
Full = gtb (ghb, magic (5)) ;
Sparse_blob = gtb_serialize (ghb, Sparse) ;
Hyper_blob  = gtb_serialize (ghb, Hyper) ;
Bitmap_blob = gtb_serialize (ghb, Bitmap) ;
Full_blob   = gtb_serialize (ghb, Full) ;
v = gtb_ver (ghb) ;

% save all matrices in gbtest_vMajor.Minor.Patch_(ghb).mat
f1 = [tempdir 'gbtest_v' v.Version '_' num2str(ghb) '.mat'] ;
fprintf ('save %s\n', f1) ;
save (f1, ...
    'Sparse', 'Hyper', 'Bitmap', 'Full', ...
    'Sparse_blob', 'Hyper_blob', 'Bitmap_blob', 'Full_blob') ;

% save just the Sparse matrix in gbtest_vMajor.Minor.Patch_save_(ghb).mat
f2 = [tempdir 'gbtest_v' v.Version '_save_' num2str(ghb) '.mat'] ;
fprintf ('save %s\n', f2) ;
gtb_save (ghb, Sparse, f2) ;

% test the current version
fprintf ('\nTesting current (v%s) mat files:\n', v.Version) ;
fprintf ('load %s\n', f1) ;
this_version = load (f1) ;
assert (isequal (Sparse, this_version.Sparse)) ;
assert (isequal (Bitmap, this_version.Bitmap)) ;
assert (isequal (Hyper , this_version.Hyper)) ;
assert (isequal (Full  , this_version.Full)) ;
S2 = gtb_deserialize (ghb, this_version.Sparse_blob) ;
B2 = gtb_deserialize (ghb, this_version.Bitmap_blob) ;
H2 = gtb_deserialize (ghb, this_version.Hyper_blob) ;
F2 = gtb_deserialize (ghb, this_version.Full_blob) ;
assert (isequal (Sparse, S2)) ;
assert (isequal (Bitmap, B2)) ;
assert (isequal (Hyper , H2)) ;
assert (isequal (Full  , F2)) ;
assert (isequal (class (Sparse), class (this_version.Sparse))) ;
assert (isequal (class (Bitmap), class (this_version.Bitmap))) ;
assert (isequal (class (Hyper) , class (this_version.Hyper))) ;
assert (isequal (class (Full)  , class (this_version.Full))) ;

fprintf ('load %s\n', f2) ;
S3 = gtb_load (ghb, f2) ;
assert (isequal (Sparse, S3)) ;

% test prior versions
fprintf ('\nTesting prior versions:\n') ;
versions = {'10.4.0', ...
    '10.3.1', '10.2.0', '10.1.1', '10.1.0', '10.0.5', ...
    '9.4.5', '9.3.1', '9.2.0', '9.1.0', '9.0.3', ...
    '8.3.1', '8.2.1', '8.0.2', ...
    '7.4.4', '7.3.3', '7.2.0', '7.1.2', '7.1.1', '7.1.0', '7.0.4', ...
    '6.2.5', '6.1.4', '6.0.2', ...
    '5.2.2', '5.1.10', '5.0.6', '5.0.2', ...
    '4.0.3', ...
    '3.3.3', '3.2.2', '3.1.1' } ;
for V = versions
    % test the prior version
    v = V {1} ;

    [major, remain] = strtok (v, '.') ;
    [minor, remain] = strtok (remain, '.') ;
    [patch, remain] = strtok (remain, '.') ;
    major = str2double (major) ;
    minor = str2double (minor) ;
    patch = str2double (patch) ;
    % fprintf ('v%d.%d.%d ', major, minor, patch) ;

    if (major > 10 || (major == 10 && minor >= 4))
        trials = 0:2 ;
    else
        trials = -1 ;
    end

    for k = trials
        if (k >= 0)
            s = ['_' num2str(k)] ;
        else
            s = '' ;
        end
        if (k <= 0)
            c = 'GrB' ;
        elseif (k == 1)
            c = 'GhB' ;
        else
            c = '(mix)' ;
        end

        f1 = ['gbtest132_matfiles/gbtest_v' v s '.mat'] ;
        fprintf ('load ./%s\n', f1) ;
        prior = load ([filepath '/' f1]) ;
        assert (isequal (Sparse, prior.Sparse)) ;
        if (k < 2)
            assert (isequal (class (prior.Sparse), c)) ;
        end

        if (isfield (prior, 'Bitmap'))
            % v3.x.x and earlier do not have bitmap format
            assert (isequal (Bitmap, prior.Bitmap)) ;
            if (k < 2)
                assert (isequal (class (prior.Bitmap), c)) ;
            end
        end

        assert (isequal (Hyper, prior.Hyper)) ;
        if (k < 2)
            assert (isequal (class (prior.Hyper), c)) ;
        end

        assert (isequal (Full, prior.Full)) ;
        if (k < 2)
            assert (isequal (class (prior.Full), c)) ;
        end

        if (isfield (prior, 'Sparse_blob'))
            % for v5.2.0 and later; v5.1.10 and earlier do not have
            % serialize/deserialize
            S2 = gtb_deserialize (ghb, prior.Sparse_blob) ;
            B2 = gtb_deserialize (ghb, prior.Bitmap_blob) ;
            H2 = gtb_deserialize (ghb, prior.Hyper_blob) ;
            F2 = gtb_deserialize (ghb, prior.Full_blob) ;
            assert (isequal (Sparse, S2)) ;
            assert (isequal (Bitmap, B2)) ;
            assert (isequal (Hyper , H2)) ;
            assert (isequal (Full  , F2)) ;
            if (ghb == 0)
                assert (isequal (class (S2), 'GrB')) ;
                assert (isequal (class (B2), 'GrB')) ;
                assert (isequal (class (H2), 'GrB')) ;
                assert (isequal (class (F2), 'GrB')) ;
            elseif (ghb == 1)
                assert (isequal (class (S2), 'GhB')) ;
                assert (isequal (class (B2), 'GhB')) ;
                assert (isequal (class (H2), 'GhB')) ;
                assert (isequal (class (F2), 'GhB')) ;
            end
        end

        if (major > 4)
            % v4 and earlier do not have GrB.load and GrB.save
            f2 = ['gbtest132_matfiles/gbtest_v' v '_save' s '.mat'] ;
            fprintf ('load ./%s\n', f2) ;
            S3 = gtb_load (ghb, [filepath '/' f2]) ;
            assert (isequal (Sparse, S3)) ;
        end
    end
    % fprintf ('\n') ;
end

fprintf ('\ngbtest132 (%d): all tests passed\n', ghb) ;

