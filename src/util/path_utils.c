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

    // resolve the full path to absolute canonical paths
    if (realpath(path, resolved_path) == NULL) {
        if (errno == ENOENT) {
            // Part of the path doesn't exist. Allow to proceed (open will fail),
            // so we don't leak existence outside the base via this check.
            return true;
        }
        return false;
    }

    // ensure the resolved_full starts with base
    return (strncmp(base, resolved_path, strlen(base)) == 0);
}
