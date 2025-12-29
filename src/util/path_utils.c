#include <string.h>
#include <limits.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>

// Validate that the user's specified path remains within
// the configuration base path.
//
// Returns true if:
// - The path resolves successfully and begins with base
// - The path component doesn't exist (ENOENT), in which case the caller will fail to open
// Returns false otherwise.
bool is_safe_path(
    const char *base,
    const char *path
) {
    char resolved_path[PATH_MAX];
    size_t base_len;
    // resolve the full path to absolute canonical paths
    if (realpath(path, resolved_path) == NULL) {
        if (errno == ENOENT) {
            // Part of the path doesn't exist. Allow to proceed (open will fail),
            // so we don't leak existence outside the base via this check.
            return true;
        }
        return false;
    }
    // ensure the resolved path starts with base and respects path boundaries
    base_len = strlen(base);
    if (strncmp(base, resolved_path, base_len) != 0) {
        return false;
    }
    
    // check that after base prefix, we have either:
    // - end of string (exact match)
    // - path separator (subdirectory)
    return (resolved_path[base_len] == '\0' || 
            resolved_path[base_len] == '/');
}
