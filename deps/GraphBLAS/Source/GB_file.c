//------------------------------------------------------------------------------
// GB_file.c: portable file I/O
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These methods provide portable open/close/lock/unlock/mkdir functions, in
// support of the JIT.  If the JIT is disabled at compile time, these functions
// do nothing.

// Windows references:
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/open-wopen
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/close
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/fdopen-wfdopen
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/mkdir-wmkdir
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/lock-file
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/unlock-file

#include "GB.h"
#include "GB_file.h"

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
        #define GB_OPEN         _open
        #define GB_CLOSE        _close
        #define GB_FDOPEN       _fdopen
        #define GB_MKDIR(path)  _mkdir (path)
        #define GB_READ_ONLY    (_O_RDONLY)
        #define GB_WRITE_ONLY   (_O_WRONLY | _O_CREAT | _O_APPEND)
        #define GB_READ_WRITE   (_O_RDWR   | _O_CREAT | _O_APPEND)
        #define GB_MODE         (_S_IREAD | _S_IWRITE)

    #else

        // POSIX
        #include <unistd.h>
        #include <dlfcn.h>
        #define GB_OPEN         open
        #define GB_CLOSE        close
        #define GB_FDOPEN       fdopen
        #define GB_MKDIR(path)  mkdir (path, S_IRWXU)
        #define GB_READ_ONLY    (O_RDONLY)
        #define GB_WRITE_ONLY   (O_WRONLY | O_CREAT | O_APPEND)
        #define GB_READ_WRITE   (O_RDWR   | O_CREAT | O_APPEND)
        #define GB_MODE         (S_IRUSR | S_IWUSR)

    #endif

#endif

//------------------------------------------------------------------------------
// GB_file_lock:  lock a file for exclusive writing
//------------------------------------------------------------------------------

// Returns true if successful, false on error

static bool GB_file_lock (FILE *fp, int fd)
{ 
    #ifdef NJIT
    {
        // JIT disabled
        return (false) ;
    }
    #elif GB_WINDOWS
    {
        // lock file for Windows
        _lock_file (fp) ;
        return (true) ;
    }
    #else
    {
        // lock file for POSIX
        struct flock lock ;
        lock.l_type = F_WRLCK ;
        lock.l_whence = SEEK_SET ;
        lock.l_start = 0 ;
        lock.l_len = 0 ;
        lock.l_pid = 0 ;
        return (fcntl (fd, F_SETLKW, &lock) == 0) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_file_unlock:  unlock a file
//------------------------------------------------------------------------------

// Returns true if successful, false on error

static bool GB_file_unlock (FILE *fp, int fd)
{ 
    #ifdef NJIT
    {
        // JIT disabled
        return (false) ;
    }
    #elif GB_WINDOWS
    {
        // unlock file for Windows
        _unlock_file (fp) ;
        return (true) ;
    }
    #else
    {
        // unlock file for POSIX
        struct flock lock ;
        lock.l_type = F_UNLCK ;
        lock.l_whence = SEEK_SET ;
        lock.l_start = 0 ;
        lock.l_len = 0 ;
        lock.l_pid = 0 ;
        return (fcntl (fd, F_SETLKW, &lock) == 0) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_file_open_and_lock:  open and lock a file for exclusive write
//------------------------------------------------------------------------------

bool GB_file_open_and_lock  // true if successful, false on error
(
    // input
    char *filename,     // full path to file to open
    // output
    FILE **fp_handle,   // file pointer of open file (NULL on error)
    int *fd_handle      // file descriptor of open file (-1 on error)
)
{ 

    #ifdef NJIT
    {
        // JIT disabled
        return (false) ;
    }
    #else
    {
        if (filename == NULL || fp_handle == NULL || fd_handle == NULL)
        { 
            // failure: inputs invalid
            return (false) ;
        }

        (*fp_handle) = NULL ;
        (*fd_handle) = -1 ;

        // open the file, creating it if it doesn't exist
        int fd = GB_OPEN (filename, GB_READ_WRITE, GB_MODE) ;
        if (fd == -1)
        { 
            // failure: file does not exist or cannot be created
            return (false) ;
        }

        // get the file pointer from the file descriptor
        FILE *fp = GB_FDOPEN (fd, "w+") ;
        if (fp == NULL)
        {
            // failure: cannot create file pointer from file descriptor
            GB_CLOSE (fd) ;
            return (false) ;
        }

        // lock the file
        if (!GB_file_lock (fp, fd))
        {
            // failure: cannot lock the file
            fclose (fp) ;
            return (false) ;
        }

        // success: file exists, is open, and is locked for writing
        (*fp_handle) = fp ;
        (*fd_handle) = fd ;
        return (true) ;
    }
    #endif
}

//------------------------------------------------------------------------------
// GB_file_unlock_and_close:  unlock and close a file
//------------------------------------------------------------------------------

bool GB_file_unlock_and_close   // true if successful, false on error
(
    // input/output
    FILE **fp_handle,       // file pointer, set to NULL on output
    int *fd_handle          // file descriptor, set to -1 on output
)
{ 

    #ifdef NJIT
    {
        // JIT disabled
        return (false) ;
    }
    #else
    {
        if (fp_handle == NULL || fd_handle == NULL)
        { 
            // failure: inputs invalid
            return (false) ;
        }

        FILE *fp = (*fp_handle) ;
        int fd = (*fd_handle) ;

        (*fp_handle) = NULL ;
        (*fd_handle) = -1 ;

        if (fp == NULL || fd < 0)
        { 
            // failure: inputs invalid
            return (false) ;
        }

        // unlock the file
        bool ok = GB_file_unlock (fp, fd) ;

        // close the file
        ok = ok && (fclose (fp) == 0) ;

        // return result
        return (ok) ;
    }
    #endif
}

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

