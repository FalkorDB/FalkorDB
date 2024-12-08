function set_malloc_debug (mdebug, new_debug_state)
%SET_MALLOC_DEBUG Turn on/off malloc debugging and mark the log.txt 
%
% set_malloc_debug (mdebug, new_debug_state)
%
% If mdebug is false, then no malloc debugging is performed,
% and the global GraphBLAS_debug is just set to false, regardless
% of new_debug_state.
%
% If mdebug is true, then the global GraphBLAS_debug flag is set
% to new_debug_state, and this action is logged in the log.txt file.

if (mdebug)
    % with malloc debugging, but allow it to be switched on or off,
    % depending on the test
    if (new_debug_state)
        debug_on
        fprintf ('================[malloc debugging turned on]============\n') ;
        fp = fopen ('log.txt', 'a') ;
        fprintf (fp, '[malloc debugging turned on]\n') ;
        fclose (fp) ;
    else
        debug_off
        fprintf ('================[malloc debugging turned off]===========\n') ;
        fp = fopen ('log.txt', 'a') ;
        fprintf (fp, '[malloc debugging turned off]\n') ;
        fclose (fp) ;
    end
else
    % no malloc debugging; ignore new_debug_state
    debug_off
end

