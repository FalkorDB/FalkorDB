function s = GB_sparsity (d)
%GB_SPARSITY a string describing the sparsity

if (d == 1)
    s = 'hypersparse' ;
elseif (d == 2)
    s = 'sparse' ;
elseif (d == 4)
    s = 'bitmap' ;
elseif (d == 8)
    s = 'full' ;
end


