function z = test_cast (x, type)
%TEST_CAST z = cast (x,type) but handle complex types
if (isequal (type, 'single') || isequal (type, 'single complex'))
    z = single (x) ;
elseif (isequal (type, 'double') || isequal (type, 'double complex'))
    z = double (x) ;
else
    z = cast (x, type) ;
end

