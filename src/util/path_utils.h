#pragma once

#include <stdbool.h>

// Validate that the user's specified path remains within the configuration base path.
// Example:
//   base: "/var/lib/FalkorDB/import"
//   path: "/../../unauthorized/access.pem"
// If the resolved path escapes the base, access is denied.
bool is_safe_path(const char *base, const char *path);
