/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

/* Hash Tables Implementation.
 *
 * This file implements in-memory hash tables with insert/del/replace/find/
 * get-random-element operations. Hash tables will auto-resize if needed
 * tables of power of two in size are used, collisions are handled by
 * chaining. See the source code for more information... :)
 *
 * Copyright (c) 2006-2012, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "mt19937-64.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#define DICT_OK 0
#define DICT_ERR 1

typedef struct dictEntry dictEntry; /* opaque */

typedef struct dict dict;

typedef struct dictType {
    uint64_t (*hashFunction)(const void *key);
    void *(*keyDup)(dict *d, const void *key);
    void *(*valDup)(dict *d, const void *obj);
    int (*keyCompare)(dict *d, const void *key1, const void *key2);
    void (*keyDestructor)(dict *d, void *key);
    void (*valDestructor)(dict *d, void *obj);
    int (*expandAllowed)(size_t moreMem, double usedRatio);
    /* Allow each dict and dictEntry to carry extra caller-defined metadata. The
     * extra memory is initialized to 0 when allocated. */
    size_t (*dictEntryMetadataBytes)(dict *d);
    size_t (*dictMetadataBytes)(void);
    /* Optional callback called after an entry has been reallocated (due to
     * active defrag). Only called if the entry has metadata. */
    void (*afterReplaceEntry)(dict *d, dictEntry *entry);
} dictType;

#define DICTHT_SIZE(exp) ((exp) == -1 ? 0 : (unsigned long)1<<(exp))
#define DICTHT_SIZE_MASK(exp) ((exp) == -1 ? 0 : (DICTHT_SIZE(exp))-1)

// fake hash function
// hash of key is simply key
static uint64_t nop_hash
(
	const void *key
) {
	return ((uint64_t)key);
}

struct dict {
    const dictType *type;

    dictEntry **ht_table[2];
    unsigned long ht_used[2];

    long rehashidx; /* rehashing not in progress if rehashidx == -1 */

    /* Keep small vars at end for optimal (minimal) struct padding */
    int16_t pauserehash; /* If >0 rehashing is paused (<0 indicates coding error) */
    signed char ht_size_exp[2]; /* exponent of size. (size = 1<<exp) */

    void *metadata[];           /* An arbitrary number of bytes (starting at a
                                 * pointer-aligned address) of size as defined
                                 * by dictType's dictEntryBytes. */
};

// global default dictType with identity hash function
dictType def_dt;

/* If safe is set to 1 this is a safe iterator, that means, you can call
 * dictAdd, dictFind, and other functions against the dictionary even while
 * iterating. Otherwise it is a non safe iterator, and only dictNext()
 * should be called while iterating. */
typedef struct dictIterator {
    dict *d;
    long index;
    int table, safe;
    dictEntry *entry, *nextEntry;
    /* unsafe iterator fingerprint for misuse detection. */
    unsigned long long fingerprint;
} dictIterator;

typedef void (dictScanFunction)(void *privdata, const dictEntry *de);
typedef void *(dictDefragAllocFunction)(void *ptr);

/* This is the initial size of every hash table */
#define DICT_HT_INITIAL_EXP      2
#define DICT_HT_INITIAL_SIZE     (1<<(DICT_HT_INITIAL_EXP))

/* ------------------------------- Macros ------------------------------------*/
#define dictFreeVal(d, entry) do {                     \
    if ((d)->type->valDestructor)                      \
        (d)->type->valDestructor((d), HashTableGetVal(entry)); \
   } while(0)

#define dictFreeKey(d, entry) \
    if ((d)->type->keyDestructor) \
        (d)->type->keyDestructor((d), HashTableGetKey(entry))

#define dictCompareKeys(d, key1, key2) \
    (((d)->type->keyCompare) ? \
        (d)->type->keyCompare((d), key1, key2) : \
        (key1) == (key2))

#define HashTableEntryMetadataSize(d) ((d)->type->dictEntryMetadataBytes     \
                                  ? (d)->type->dictEntryMetadataBytes(d) : 0)
#define dictMetadataSize(d) ((d)->type->dictMetadataBytes               \
                             ? (d)->type->dictMetadataBytes() : 0)

#define dictHashKey(d, key) ((d)->type->hashFunction(key))
#define dictSlots(d) (DICTHT_SIZE((d)->ht_size_exp[0])+DICTHT_SIZE((d)->ht_size_exp[1]))
#define dictSize(d) ((d)->ht_used[0]+(d)->ht_used[1])
#define dictIsRehashing(d) ((d)->rehashidx != -1)
#define dictPauseRehashing(d) ((d)->pauserehash++)
#define dictResumeRehashing(d) ((d)->pauserehash--)

/* If our unsigned long type can store a 64 bit number, use a 64 bit PRNG. */
#if ULONG_MAX >= 0xffffffffffffffff
#define randomULong() ((unsigned long) genrand64_int64())
#else
#define randomULong() random()
#endif

typedef enum {
    DICT_RESIZE_ENABLE,
    DICT_RESIZE_AVOID,
    DICT_RESIZE_FORBID,
} HashTableResizeEnable;

/* API */
dict *HashTableCreate(const dictType *type);
unsigned long HashTableElemCount(const dict *d);
int HashTableExpand(dict *d, unsigned long size);
void *HashTableMetadata(dict *d);
int HashTableAdd(dict *d, void *key, void *val);
dictEntry *HashTableAddRaw(dict *d, void *key, dictEntry **existing);
dictEntry *HashTableAddOrFind(dict *d, void *key);
int HashTableReplace(dict *d, void *key, void *val);
int HashTableDelete(dict *d, const void *key);
dictEntry *HashTableUnlink(dict *d, const void *key);
void HashTableFreeUnlinkedEntry(dict *d, dictEntry *he);
dictEntry *HashTableTwoPhaseUnlinkFind(dict *d, const void *key, dictEntry ***plink, int *table_index);
void HashTableTwoPhaseUnlinkFree(dict *d, dictEntry *he, dictEntry **plink, int table_index);
void HashTableRelease(dict *d);
dictEntry *HashTableFind(dict *d, const void *key);
void *HashTableFetchValue(dict *d, const void *key);
int HashTableResize(dict *d);
void HashTableSetKey(dict *d, dictEntry* de, void *key);
void HashTableSetVal(dict *d, dictEntry *de, void *val);
void *HashTableEntryMetadata(dictEntry *de);
void *HashTableGetKey(const dictEntry *de);
void *HashTableGetVal(const dictEntry *de);
size_t HashTableMemUsage(const dict *d);
size_t HashTableEntryMemUsage(void);
dictIterator *HashTableGetIterator(dict *d);
dictIterator *HashTableGetSafeIterator(dict *d);
void HashTableInitIterator(dictIterator *iter, dict *d);
void HashTableInitSafeIterator(dictIterator *iter, dict *d);
void HashTableResetIterator(dictIterator *iter);
dictEntry *HashTableNext(dictIterator *iter);
void HashTableReleaseIterator(dictIterator *iter);
void HashTableGetStats(char *buf, size_t bufsize, dict *d);
uint64_t HashTableGenHashFunction(const void *key, size_t len);
uint64_t HashTableGenCaseHashFunction(const unsigned char *buf, size_t len);
void HashTableEmpty(dict *d, void(callback)(dict*));
void HashTableSetResizeEnabled(HashTableResizeEnable enable);
int HashTableRehash(dict *d, int n);
int HashTableRehashMilliseconds(dict *d, int ms);
void HashTableSetHashFunctionSeed(uint8_t *seed);
uint8_t *HashTableGetHashFunctionSeed(void);
unsigned long HashTableScan(dict *d, unsigned long v, dictScanFunction *fn, void *privdata);
unsigned long HashTableScanDefrag(dict *d, unsigned long v, dictScanFunction *fn, dictDefragAllocFunction *allocfn, void *privdata);
uint64_t HashTableGetHash(dict *d, const void *key);
dictEntry *HashTableFindEntryByPtrAndHash(dict *d, const void *oldptr, uint64_t hash);

