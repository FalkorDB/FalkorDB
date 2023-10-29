# - Find check
# Find the native check headers and libraries.
#
# CHECK_INCLUDE_DIRS	- where to find check.h, etc.
# CHECK_LIBRARIES	- List of libraries when using check.
# CHECK_FOUND	- True if check has been found.

# Look for the header file.
find_path (CHECK_INCLUDE_DIR check.h)

# Look for the library.
find_library (CHECK_LIBRARY NAMES check)
find_library (COMPAT_LIBRARY NAMES compat)

# Handle the QUIETLY and REQUIRED arguments and set CHECK_FOUND to TRUE if all listed variables are TRUE.
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (CHECK DEFAULT_MSG CHECK_LIBRARY CHECK_INCLUDE_DIR)

# Copy the results to the output variables.
if (CHECK_FOUND)
    set (CHECK_LIBRARIES ${CHECK_LIBRARY} ${COMPAT_LIBRARY})
    set (CHECK_INCLUDE_DIRS ${CHECK_INCLUDE_DIR})
else (CHECK_FOUND)
    set (CHECK_LIBRARIES)
    set (CHECK_INCLUDE_DIRS)
endif (CHECK_FOUND)

mark_as_advanced (CHECK_INCLUDE_DIRS CHECK_LIBRARIES)
