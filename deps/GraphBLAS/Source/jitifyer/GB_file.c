//------------------------------------------------------------------------------
// GB_file.c: portable file I/O
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These methods provide portable file I/O functions in support of the JIT.  If
// the JIT is disabled at compile time, these functions do nothing.

// Windows references:
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/mkdir-wmkdir

#include "GB.h"
#include "jitifyer/GB_file.h"

#ifndef NJIT

    #include <fcntl.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <errno.h>

    #if GB_WINDOWS

        // Windows
        #include <io.h>
        #include <direct.h>
        #include <windows.h>
        #define GB_MKDIR(path)  _mkdir (path)

    #else

        // POSIX
        #include <unistd.h>
        #include <dlfcn.h>
        #define GB_MKDIR(path)  mkdir (path, S_IRWXU)

    #endif

#endif

//------------------------------------------------------------------------------
// GB_file_mkdir: create a directory
//------------------------------------------------------------------------------

// Create a directory, including all parent directories if they do not exist.
// Returns true if the directory already exists or if it was successfully
// created.  Returns true if the JIT is disabled (the directory is not created
// but also not needed in that case).  Returns false on error.

bool GB_file_mkdir (char *path)
{
    if (path == NULL)
    { 
        // invalid input
        return (false) ;
    }

    #ifdef NJIT
    {
        // JIT disabled; do not create the directory but do not return an error
        return (true) ;
    }
    #else
    {
        // create all the leading directories
        int result = 0 ;
        bool first = true ;
        for (char *p = path ; *p ; p++)
        {
            // look for a file separator
            if (*p == '/' || *p == '\\')
            {
                // found a file separator
                if (!first)
                { 
                    // terminate the path at this file separator
                    char save = *p ;
                    *p = '\0' ;
                    // construct the directory at this path
                    result = GB_MKDIR (path) ;
                    // err = (result == -1) ? errno : 0 ;
                    // restore the path
                    *p = save ;
                }
                first = false ;
            }
        }

        // create the final directory
        result = GB_MKDIR (path) ;
        int err = (result == -1) ? errno : 0 ;
        return (err == 0 || err == EEXIST) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_file_dlopen: open a dynamic library
//------------------------------------------------------------------------------

void *GB_file_dlopen (char *library_name)
{ 
    #ifdef NJIT
    {
        // JIT disabled
        return (NULL) ;
    }
    #elif GB_WINDOWS
    {
        // open a Windows dll
        HINSTANCE hdll = LoadLibrary (library_name) ;
        return ((void *) hdll) ;
    }
    #else
    {
        // open a POSIX dynamic library
        return (dlopen (library_name, RTLD_LAZY)) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_file_dlsym: get a function pointer from a dynamic library
//------------------------------------------------------------------------------

void *GB_file_dlsym (void *dl_handle, char *symbol)
{ 
    #ifdef NJIT
    {
        // JIT disabled
        return (NULL) ;
    }
    #elif GB_WINDOWS
    {
        // lookup a symbol in a Windows dll
        void *p = (void *) GetProcAddress (dl_handle, symbol) ;
        return ((void *) p) ;
    }
    #else
    {
        // lookup a symbol in a POSIX dynamic library
        return (dlsym (dl_handle, symbol)) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_file_dlclose: close a dynamic library
//------------------------------------------------------------------------------

void GB_file_dlclose (void *dl_handle)
{ 
    if (dl_handle != NULL)
    {
        #ifdef NJIT
        {
            // JIT disabled: do nothing
        }
        #elif GB_WINDOWS
        {
            // close a Windows dll
            FreeLibrary (dl_handle) ;
        }
        #else
        {
            // close a POSIX dynamic library
            dlclose (dl_handle) ;
        }
        #endif
    }
}

