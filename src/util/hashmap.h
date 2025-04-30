// Copyright 2020 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

#ifndef HASHMAP_H
#define HASHMAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

typedef struct hashmap *hashmap;

hashmap hashmap_new(size_t elsize, size_t cap, uint64_t seed0, 
    uint64_t seed1, 
    uint64_t (*hash)(const void *item, uint64_t seed0, uint64_t seed1),
    int (*compare)(const void *a, const void *b, void *udata),
    void (*elfree)(void *item),
    void *udata);

hashmap hashmap_new_with_allocator(void *(*malloc)(size_t), 
    void *(*realloc)(void *, size_t), void (*free)(void*), size_t elsize, 
    size_t cap, uint64_t seed0, uint64_t seed1,
    uint64_t (*hash)(const void *item, uint64_t seed0, uint64_t seed1),
    int (*compare)(const void *a, const void *b, void *udata),
    void (*elfree)(void *item),
    void *udata);

hashmap hashmap_new_with_redis_allocator(size_t elsize, 
    size_t cap, uint64_t seed0, uint64_t seed1,
    uint64_t (*hash)(const void *item, uint64_t seed0, uint64_t seed1),
    int (*compare)(const void *a, const void *b, void *udata),
    void (*elfree)(void *item),
    void *udata);

void hashmap_free(hashmap map);
void hashmap_clear(hashmap map, bool update_cap);
size_t hashmap_count(hashmap map);
bool hashmap_oom(hashmap map);
const void *hashmap_get(hashmap map, const void *item);
const void *hashmap_set(hashmap map, const void *item);
const void *hashmap_delete(hashmap map, const void *item);
const void *hashmap_probe(hashmap map, uint64_t position);
bool hashmap_scan(hashmap map, bool (*iter)(const void *item, void *udata), void *udata);
bool hashmap_iter(hashmap map, size_t *i, void **item);

uint64_t hashmap_sip(const void *data, size_t len, uint64_t seed0, uint64_t seed1);
uint64_t hashmap_murmur(const void *data, size_t len, uint64_t seed0, uint64_t seed1);
uint64_t hashmap_xxhash3(const void *data, size_t len, uint64_t seed0, uint64_t seed1);

const void *hashmap_get_with_hash(hashmap map, const void *key, uint64_t hash);
const void *hashmap_delete_with_hash(hashmap map, const void *key, uint64_t hash);
const void *hashmap_set_with_hash(hashmap map, const void *item, uint64_t hash);
void hashmap_set_grow_by_power(hashmap map, size_t power);
void hashmap_set_load_factor(hashmap map, double load_factor);


// DEPRECATED: use `hashmap_new_with_allocator`
void hashmap_set_allocator(void *(*malloc)(size_t), void (*free)(void*));

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // HASHMAP_H
