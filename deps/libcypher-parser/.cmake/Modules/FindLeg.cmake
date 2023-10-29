# - FindLeg leg
# Find the native leg program.
#
# LEG_PROGRAMS - List of programs when using leg.
# LEG_FOUND	- True if leg has been found.

# Look for the library.
find_program (LEG_PROGRAM NAMES leg)

# Handle the QUIETLY and REQUIRED arguments and set LEG_FOUND to TRUE if all listed variables are TRUE.
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (LEG DEFAULT_MSG LEG_PROGRAM)

# Copy the results to the output variables.
if (LEG_FOUND)
    set (LEG_PROGRAMS ${LEG_PROGRAM})
else (LEG_FOUND)
    set (LEG_PROGRAMS)
endif (LEG_FOUND)

mark_as_advanced (LEG_PROGRAMS)
