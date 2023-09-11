# - Find getopt
# Find the native getopt headers and libraries.
#
# GETOPT_INCLUDE_DIRS	- where to find getopt.h, etc.
# GETOPT_LIBRARIES	- List of libraries when using getopt.
# GETOPT_FOUND	- True if getopt has been found.

# Look for the header file.
find_path (GETOPT_INCLUDE_DIR getopt.h)

# Look for the library.
find_library (GETOPT_LIBRARY NAMES wingetopt)

# Handle the QUIETLY and REQUIRED arguments and set GETOPT_FOUND to TRUE if all listed variables are TRUE.
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (GETOPT DEFAULT_MSG GETOPT_LIBRARY GETOPT_INCLUDE_DIR)

# Copy the results to the output variables.
if (GETOPT_FOUND)
    set (GETOPT_LIBRARIES ${GETOPT_LIBRARY})
    set (GETOPT_INCLUDE_DIRS ${GETOPT_INCLUDE_DIR})
else (GETOPT_FOUND)
    set (GETOPT_LIBRARIES)
    set (GETOPT_INCLUDE_DIRS)
endif (GETOPT_FOUND)

mark_as_advanced (GETOPT_INCLUDE_DIRS GETOPT_LIBRARIES)
