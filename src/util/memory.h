/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// returns the current Resident Set Size (RSS) in bytes
size_t get_current_rss(void) ;

// returns the total amount of available physical memory on the host in bytes
size_t get_host_available_memory(void) ;

