% tsv_to_mtx.m:  converts the original tsv files for the smallest sparse
% deep neural network problem at https://graphchallenge.mit.edu/data-sets
% to a test case for LAGraph_dnn.  Only the first 30 layers are used out
% of the original 120, and only the first 1200 features are used out of
% the original 60,000.

try
    delete ('comments.txt')
catch me
end

nfeatures = 60000 ;
nfeatures_subset = 1200 ;
nneurons = 1024 ;
nlayers = 1920 ;
nlayers_subset = 30 ;

for k = 120 % [120 480 1920]
    infile  = sprintf ('/Users/davis/dnn_data/DNN/neuron1024-l%d-categories.tsv', k) ;
    outfile = sprintf ('neuron1024-l%d-categories_subset.mtx', k) ;
    T = load (infile) ;
    ncat = length (T) ;
    fprintf ('ncat %d\n', ncat) ;
    T (:,2) = ones (ncat, 1) ;
    A = logical (sparse (T (:,1), T (:,2), true, nfeatures, 1)) ;
    A = A (1:nfeatures_subset, 1) ;
    fc = fopen ('comments.txt', 'w+') ;
    fprintf (fc, '%%GraphBLAS type bool\n') ;
    fprintf (fc, ' Synthetic Sparse Deep Neural Network, output categories\n') ;
    fprintf (fc, ' nneurons: %d, nlayers: %d, nfeatures: %d subset: %d\n', ...
        nneurons, nlayers, nfeatures, nfeatures_subset) ;
    fprintf (fc, ' Source: https://graphchallenge.mit.edu/data-sets\n') ;
    fprintf (fc, ' The original problem has %d features but this\n', nfeatures) ;
    fprintf (fc, ' subset only includes the first %d.\n', nfeatures_subset) ;
    fprintf (fc, ' Converted to Matrix Matrix Format by Tim Davis.\n') ;
    mwrite (outfile, A, 'comments.txt') ;
    delete ('comments.txt')
end

C = cell (nlayers_subset,1) ;
for k = 1:nlayers_subset % 1:nlayers
    infile  = sprintf ('/Users/davis/dnn_data/DNN/neuron1024/n1024-l%d.tsv', k);
    outfile = sprintf ('n1024-l%d.mtx', k) ;
    T = load (infile) ;
    A = sparse (T (:,1), T (:,2), T (:,3), nneurons, nneurons) ;
    C {k} = A ;
    fprintf ('%d: nnz %d\n', k, nnz (A)) ;
    if (nnz (A) ~= size (T,1))
        error ('duplicates!\n') ;
    end
    fc = fopen ('comments.txt', 'w+') ;
    fprintf (fc, '%%GraphBLAS type float\n') ;
    fprintf (fc, ' Synthetic Sparse Deep Neural Network, layer %d of %d.\n', k, nlayers) ;
    fprintf (fc, ' Source: https://graphchallenge.mit.edu/data-sets\n') ;
    fprintf (fc, ' Converted to Matrix Matrix Format by Tim Davis.\n') ;
    fclose (fc) ;
    mwrite (outfile, A, 'comments.txt') ;
    delete ('comments.txt')
end

infile = sprintf ('/Users/davis/dnn_data/MNIST/sparse-images-1024.tsv') ;
outfile = sprintf ('sparse-images-1024_subset.mtx') ;
T = load (infile) ;
fc = fopen ('comments.txt', 'w+') ;
fprintf (fc, '%%GraphBLAS type float\n') ;
fprintf (fc, ' Synthetic Sparse Deep Neural Network, Input images.\n') ;
fprintf (fc, ' Source: https://graphchallenge.mit.edu/data-sets\n') ;
fprintf (fc, ' The original problem has %d features but this\n', nfeatures) ;
fprintf (fc, ' subset only includes the first %d.\n', nfeatures_subset) ;
fprintf (fc, ' Converted to Matrix Matrix Format by Tim Davis.\n') ;
fclose (fc) ;
A = sparse (T (:,1), T (:,2), T (:,3), nfeatures, nneurons) ;
A = A (1:nfeatures_subset, :) ;
mwrite (outfile, A, 'comments.txt') ;
fprintf ('images nnz %d\n', nnz (A)) ;
delete ('comments.txt')

if (0)
    for k = 1:1920
        for t = k+1:1920
            if (isequal (C {k}, C {t}))
                fprintf ('%d == %d\n', t, k) ;
            end
        end
    end
end

