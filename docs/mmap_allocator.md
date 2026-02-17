# MMAP-Backed Memory Allocator for FalkorDB

## Overview

This implementation adds an optional mmap-backed memory allocator to FalkorDB that allows memory allocations to be backed by file-mapped memory, enabling the system to extend beyond available RAM when needed.

## Design

### Architecture

The mmap allocator uses a **chunk-based allocation strategy**:

1. **Small Allocations (< chunk_size)**: Allocated from pre-allocated mmap chunks
   - Default chunk size: 100MB (`MMAP_DEFAULT_CHUNK_SIZE`)
   - Chunks are allocated on-demand when current chunk is full
   - Uses a simple bump allocator within each chunk for fast allocation
   - Memory is reused when chunks are freed (no per-allocation freeing within chunks)

2. **Large Allocations (>= chunk_size)**: Use dedicated mmap regions
   - Each large allocation gets its own mmap region
   - These can be individually freed and unmapped
   - Useful for large graph structures or result sets

### Key Features

- **Thread-safe**: Uses pthread mutex for chunk list management
- **Atomic statistics**: Lock-free statistics tracking using C11 atomics
- **16-byte alignment**: All allocations are aligned to 16 bytes
- **Memory tracking**: Compatible with existing memory capacity limits
- **Optional**: Can be enabled/disabled via configuration

### Files

- `src/util/mmap_alloc.h` - Header file with API definitions
- `src/util/mmap_alloc.c` - Implementation of the mmap allocator
- `src/util/rmalloc.h` - Updated to include mmap allocator integration
- `src/util/rmalloc.c` - Updated to use mmap allocator when enabled
- `tests/unit/test_mmap_alloc.c` - Comprehensive unit tests

## API

### Initialization

```c
// Initialize allocator with default 100MB chunks
bool rm_mmap_init(0);

// Initialize with custom chunk size
bool rm_mmap_init(50 * 1024 * 1024); // 50MB chunks
```

### Cleanup

```c
// Free all mmap allocator resources
void rm_mmap_cleanup(void);
```

### Statistics

```c
mmap_stats_t stats;
rm_mmap_get_stats(&stats);

// stats.chunk_size - configured chunk size
// stats.chunks_allocated - number of chunks allocated
// stats.large_allocs - number of large allocations
// stats.total_memory - total memory allocated
// stats.memory_in_use - memory currently in use
```

## Integration with rmalloc

The mmap allocator is integrated into the existing `rmalloc` memory management system. When enabled:

- `rm_malloc()` uses `mmap_alloc()`
- `rm_calloc()` uses `mmap_alloc()` + memset
- `rm_realloc()` uses `mmap_realloc()`
- `rm_free()` uses `mmap_free()`
- `rm_strdup()` uses `mmap_alloc()` + memcpy

Memory tracking and capacity limits continue to work as before.

## Configuration

Two new configuration options are added:

- `Config_MMAP_ALLOCATOR_ENABLED` - Enable/disable the mmap allocator (boolean)
- `Config_MMAP_ALLOCATOR_CHUNK_SIZE` - Set chunk size in bytes (uint64)

These can be set at module load time or changed at runtime via GRAPH.CONFIG SET.

## Usage Example

The mmap allocator is designed to be transparent to the rest of the codebase. To enable it:

1. Initialize at module load time in `module.c`:
```c
// Initialize mmap allocator with 100MB chunks
rm_mmap_init(0);
```

2. All existing code using `rm_malloc()`, `rm_free()`, etc. will automatically use the mmap allocator.

3. Clean up at module unload:
```c
rm_mmap_cleanup();
```

## Memory Management

### Chunk Allocation

- Chunks are allocated using `mmap()` with `MAP_PRIVATE | MAP_ANONYMOUS`
- New chunks are added to the front of the linked list
- Chunks are never freed individually (only at cleanup)

### Large Allocations

- Allocations >= chunk_size bypass the chunk system
- Use dedicated `mmap()` for the exact size needed
- Can be freed individually with `munmap()`

### Deallocation

- Small allocations: Memory tracking is updated but chunk memory is not freed
- Large allocations: Immediately unmapped with `munmap()`
- At cleanup: All chunks and remaining large allocations are freed

## Performance Considerations

### Advantages

- Allows memory to be backed by swap when physical RAM is full
- Fast allocation for small objects (bump allocator)
- No per-allocation overhead for small allocations within chunks
- Large allocations can be freed individually

### Trade-offs

- Chunk memory is not reclaimed until cleanup
- May use more total memory due to chunk pre-allocation
- Small allocation freeing doesn't actually release memory

### Recommendations

- Use default 100MB chunks for general purpose workloads
- Increase chunk size for workloads with many large allocations
- Decrease chunk size if memory is constrained
- Monitor statistics to tune chunk size appropriately

## Testing

Comprehensive unit tests are provided in `tests/unit/test_mmap_alloc.c`:

- Initialization and cleanup
- Small allocations from chunks
- Large allocations (> chunk size)
- Multiple chunks
- Realloc functionality
- Edge cases (NULL, zero size, etc.)
- Memory alignment

Run tests with:
```bash
make unit-tests
```

## Limitations

1. **No per-allocation freeing within chunks**: Small allocations don't release memory until chunk cleanup
2. **Chunk pre-allocation**: May allocate more memory than immediately needed
3. **Platform-specific**: Uses POSIX mmap, requires Linux/Unix

## Future Enhancements

Possible improvements:

1. **Free list allocator**: Support per-allocation freeing within chunks
2. **Chunk recycling**: Reuse empty chunks instead of keeping all chunks
3. **NUMA awareness**: Allocate chunks on specific NUMA nodes
4. **File-backed mmap**: Support actual file backing for persistence
5. **Huge pages**: Use huge pages for better TLB efficiency

## Implementation Notes

- The allocator is NOT enabled by default - it must be explicitly initialized
- Compatible with existing memory tracking and capacity limits
- Thread-safe for concurrent allocations from multiple threads
- Statistics are lock-free using C11 atomics
- All allocations include a header with size and metadata

## Migration Path

For existing code:

1. No changes needed to allocation calls (`rm_malloc`, `rm_free`, etc.)
2. Add initialization call at module startup
3. Add cleanup call at module shutdown
4. Optional: Add configuration to enable/disable at runtime

The mmap allocator is a drop-in replacement for the existing allocator.
