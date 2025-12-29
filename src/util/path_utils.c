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
    char *canonical_base = realpath(base, NULL);
    if (canonical_base == NULL) {
        // base path must exist and be resolvable
        return false;
    }
    // Remove trailing slash except for root
    size_t canon_len = strlen(canonical_base);
    if (canon_len > 1 && canonical_base[canon_len - 1] == '/') {
        canonical_base[canon_len - 1] = '\0';
        canon_len--;
    }
    // resolve the full path to absolute canonical paths
    if (realpath(path, resolved_path) == NULL) {
        free(canonical_base);
        if (errno == ENOENT) {
            // Part of the path doesn't exist. Allow to proceed (open will fail),
            // so we don't leak existence outside the base via this check.
            return true;
        }
        return false;
    }
    // ensure the resolved path starts with canonical_base and respects path boundaries
    if (strncmp(canonical_base, resolved_path, canon_len) != 0) {
        free(canonical_base);
        return false;
    }
    // check that after base prefix, we have either:
    // - end of string (exact match)
    // - path separator (subdirectory)
    bool safe = (resolved_path[canon_len] == '\0' || resolved_path[canon_len] == '/');
    free(canonical_base);
    return safe;
}
