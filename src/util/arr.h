#ifndef UTIL_ARR_H_
#define UTIL_ARR_H_
/* arr.h - simple, easy to use dynamic array with fat pointers,
 * to allow native access to members. It can accept pointers, struct literals and scalars.
 *
 * Example usage:
 *
 *  int *arr = array_new(int, 8);
 *  // Add elements to the array
 *  for (int i = 0; i < 100; i++) {
 *   array_append(arr, i);
 *  }
 *
 *  // read individual elements
 *  for (int i = 0; i < array_len(arr); i++) {
 *    printf("%d\n", arr[i]);
 *  }
 *
 *  array_free(arr);
 *
 *
 *  */
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include "../RG.h"
#include "rmalloc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Definition of malloc & friends that can be overridden before including arr.h.
// Alternatively you can include arr_rm_alloc.h, which wraps arr.h and sets the
// allocation functions to those of the RM_ family

#ifndef array_alloc_fn
#define array_alloc_fn rm_malloc
#define array_realloc_fn rm_realloc
#define array_free_fn rm_free
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

typedef struct {
	uint32_t len;
	// TODO: optimize memory by making cap a 16-bit delta from len, and elem_sz 16 bit as well. This
	// makes the whole header fit in 64 bit
	uint32_t cap;
	uint32_t elem_sz;
	char buf[];
} array_hdr_t;

#define arrayof(T) T *

typedef void *array_t;
/* Internal - calculate the array size for allocations */
#define array_sizeof(hdr) (sizeof(array_hdr_t) + (uint64_t)hdr->cap * hdr->elem_sz)
/* Internal - get a pointer to the array header */
#define array_hdr(arr) ((array_hdr_t *)(((char *)arr) - sizeof(array_hdr_t)))
/* Internal - get a pointer to an element inside the array at a given index */
#define array_elem(arr, idx) ((void *)((char *)arr + (idx * array_hdr(arr)->elem_sz)))

static inline uint32_t array_len(array_t arr);

/* Initialize a new array with a given element size and capacity. Should not be used directly - use
 * array_new instead */
static array_t array_new_sz(uint32_t elem_sz, uint32_t cap, uint32_t len) {
	uint64_t array_size = (uint64_t)cap * elem_sz;
	array_hdr_t *hdr = (array_hdr_t *)array_alloc_fn(sizeof(array_hdr_t) + array_size);
	hdr->cap = cap;
	hdr->elem_sz = elem_sz;
	hdr->len = len;
	return (array_t)(hdr->buf);
}

/* Functions declared as symbols for use in debugger */
void array_debug(void *pp);

/* Initialize an array for a given type T with a given capacity and zero length. The array should be
 * case to a pointer to that type. e.g.
 *
 *  int *arr = array_new(int, 4);
 *
 * This allows direct access to elements
 *  */
#define array_new(T, cap) (T *)(array_new_sz(sizeof(T), cap, 0))

/* initialize an array for a given type T with a given length
 * the capacity allocated is identical
 * to the length
 *  */
#define array_newlen(T, len) (T *)(array_new_sz(sizeof(T), len, len))

static inline array_t array_ensure_cap(array_t arr, uint32_t cap) {
	array_hdr_t *hdr = array_hdr(arr);
	if(cap > hdr->cap) {
		hdr->cap = MAX(hdr->cap * 2, cap);
		hdr = (array_hdr_t *)array_realloc_fn(hdr, array_sizeof(hdr));
	}
	return (array_t)hdr->buf;
}

// Reallocates the array setting the new capacity. If there were more
// elements, those are deleted (cut off), if less, then there will be
// space enough to fit (capacity - length) elements.
static inline array_t array_reset_cap(array_t arr, const uint32_t cap) {
  array_hdr_t *hdr = array_hdr(arr);
  hdr->cap = cap;
  hdr->len = hdr->len > hdr->cap ? hdr->cap : hdr->len;
  hdr = (array_hdr_t *)array_realloc_fn(hdr, array_sizeof(hdr));
	return (array_t)hdr->buf;
}

/* Ensure capacity for the array to grow by one */
static inline array_t array_grow(array_t arr, size_t n) {
	array_hdr(arr)->len += n;
	return array_ensure_cap(arr, array_hdr(arr)->len);
}

static inline array_t array_ensure_len(array_t arr, size_t len) {
	if(len <= array_len(arr)) {
		return arr;
	}
	len -= array_len(arr);
	return array_grow(arr, len);
}

/* Ensures that array_tail will always point to a valid element. */
#define array_ensure_tail(arrpp, T)            \
  ({                                           \
    if (!*(arrpp)) {                           \
      *(arrpp) = array_newlen(T, 1);           \
    } else {                                   \
      *(arrpp) = (T *)array_grow(*(arrpp), 1); \
    }                                          \
    &(array_tail(*(arrpp)));                   \
  })

/**
 * Appends elements to the end of the array, creating the array if it does
 * not exist
 * @param arrpp array pointer. Can be NULL
 * @param src array (i.e. C array) of elements to append
 * @param n length of sec
 * @param T type of the array (for sizeof)
 * @return the array
 */
#define array_ensure_append(arrpp, src, n, T) do {    \
    size_t a__oldlen = 0;                             \
    if (!arrpp) {                                     \
      (arrpp) = array_newlen(T, n);                   \
    } else {                                          \
      a__oldlen = array_len(arrpp);                   \
      (arrpp) = (T *)array_grow(arrpp, n);            \
    }                                                 \
    memcpy((arrpp) + a__oldlen, src, n * sizeof(T));  \
  } while(0)

/**
 * Does the same thing as ensure_append, but the added elements are
 * at the _beginning_ of the array
 */
#define array_ensure_prepend(arrpp, src, n, T)                          \
  ({                                                                    \
    size_t a__oldlen = 0;                                               \
    if (!arrpp) {                                                       \
      arrpp = array_newlen(T, n);                                       \
    } else {                                                            \
      a__oldlen = array_len(arrpp);                                     \
      arrpp = (T *)array_grow(arrpp, n);                                \
    }                                                                   \
    memmove(((char *)arrpp) + sizeof(T), arrpp, a__oldlen * sizeof(T)); \
    memcpy(arrpp, src, n * sizeof(T));                                  \
    arrpp;                                                              \
  })

/*
 * This macro is useful for sparse arrays. It ensures that `*arrpp` will
 * point to a valid index in the array, growing the array to fit.
 *
 * If the array needs to be expanded in order to contain the index, then
 * the unused portion of the array (i.e. the space between the previously
 * last-valid element and the new index) is zero'd
 *
 * @param arrpp a pointer to the array (e.g. `T**`)
 * @param pos the index that should be considered valid
 * @param T the type of the array (in case it must be created)
 * @return A pointer of T at the requested index
 */
#define array_ensure_at(arrpp, pos, T)                                    \
  ({                                                                      \
    if (!(*arrpp)) {                                                      \
      *(arrpp) = array_new(T, 1);                                         \
    }                                                                     \
    if (array_len(*arrpp) <= pos) {                                       \
      size_t curlen = array_len(*arrpp);                                  \
      array_hdr(*arrpp)->len = pos + 1;                                   \
      *arrpp = (T *)array_ensure_cap(*(arrpp), array_hdr(*(arrpp))->len); \
      memset((T *)*arrpp + curlen, 0, sizeof(T) * ((pos + 1) - curlen));  \
    }                                                                     \
    (T *)(*arrpp) + pos;                                                  \
  })

/* get the last element in the array */
#define array_tail(arr) ((arr)[array_hdr(arr)->len - 1])

/* Append an element to the array, returning the array which may have been reallocated */
#define array_append(arr, x) do {                      \
    (arr) = (__typeof__(arr))array_grow((arr), 1);     \
    array_tail((arr)) = (x);                           \
  } while(0)

/* Get the length of the array */
static inline uint32_t array_len(const array_t arr) {
	return arr ? array_hdr(arr)->len : 0;
}

#define ARR_CAP_NOSHRINK ((uint32_t)-1)
static inline void *array_trimm(array_t arr, uint32_t len, uint32_t cap) {
	array_hdr_t *arr_hdr = array_hdr(arr);
	ASSERT((cap == ARR_CAP_NOSHRINK || cap > 0 || len == cap) && "trimming capacity is illegal");
	ASSERT((cap == ARR_CAP_NOSHRINK || cap >= len) && "trimming len is greater then capacity");
	ASSERT((len <= arr_hdr->len) && "trimming len is greater then current len");
	arr_hdr->len = len;
	if(cap != ARR_CAP_NOSHRINK) {
		arr_hdr->cap = cap;
		arr_hdr = (array_hdr_t *)array_realloc_fn(arr_hdr, array_sizeof(arr_hdr));
	}
	return arr_hdr->buf;
}

#define array_trimm_len(arr, len) (__typeof__(arr)) array_trimm(arr, len, ARR_CAP_NOSHRINK)
#define array_trimm_cap(arr, len) (__typeof__(arr)) array_trimm(arr, len, len)

/* Free the array, without dealing with individual elements */
static void array_free(array_t arr) {
	if(arr != NULL) {
		// like free(), shouldn't explode if NULL
		array_free_fn(array_hdr(arr));
	}
}

#define array_clear(arr) array_hdr(arr)->len = 0

/* Repeate the code in "blk" for each element in the array, and give it the name of "as".
 * e.g:
 *  int *arr = array_new(int, 10);
 *  array_append(arr, 1);
 *  array_foreach(arr, i, printf("%d\n", i));
 */
#define array_foreach(arr, as, blk)                 \
  ({                                                \
    for (uint32_t i = 0; i < array_len(arr); i++) { \
      __typeof__(*arr) as = arr[i];                 \
      blk;                                          \
    }                                               \
  })

/* Free the array, free individual element using callback */
#define array_free_cb(arr, cb)                        \
  ({                                                  \
    if (arr) {                                        \
      for (uint32_t i = 0; i < array_len(arr); i++) { \
        { cb(arr[i]); }                               \
      }                                               \
      array_free(arr);                                \
    }                                                 \
  })

/* Pop the top element from the array, reduce the size and return it */
#define array_pop(arr)               \
  __extension__ ({                   \
    ASSERT(array_hdr(arr)->len > 0); \
    (arr)[--(array_hdr(arr)->len)];  \
  })

/* Remove a specified element from the array */
#define array_del(arr, ix)                                                         \
  __extension__({                                                                  \
    ASSERT(array_len(arr) > ix);                                                   \
    if (array_len(arr) - 1 > ix) {                                                 \
      memmove(arr + ix, arr + ix + 1, sizeof(*arr) * (array_len(arr) - (ix + 1))); \
    }                                                                              \
    --array_hdr(arr)->len;                                                         \
    arr;                                                                           \
  })

/* Remove a specified element from the array, but does not preserve order */
#define array_del_fast(arr, ix)                \
  __extension__({                              \
    if (array_len((arr)) > 1) {                \
      (arr)[ix] = (arr)[array_len((arr)) - 1]; \
    }                                          \
    --array_hdr((arr))->len;                   \
    arr;                                       \
  })

/* Duplicate the array to the pointer dest. */
#define array_clone(dest, arr)                            \
  __extension__({                                         \
   dest = array_newlen(typeof(*arr), array_len(arr));     \
   memcpy(dest, arr, sizeof(*arr) * (array_len(arr)));    \
  })

/* Duplicate an array with a dedicated value clone callback. */
#define array_clone_with_cb(dest, arr, clone_cb)        \
__extension__({                                         \
    uint arrayLen = array_len((arr));                   \
    dest = array_new(__typeof__(*arr), arrayLen);       \
    for(uint i = 0; i < arrayLen; i++)                  \
        array_append(dest, (clone_cb(arr[i])));  \
})

#define array_reverse(arr)                      \
    __extension__({                             \
        uint arrayLen = array_len(arr);         \
        for(uint i = 0; i < arrayLen/2; i++) {  \
            __typeof__(*arr) tmp = arr[i];      \
            uint j = arrayLen -1 -i;            \
            arr[i] = arr[j];                    \
            arr[j] = tmp;                       \
        }                                       \
    })                                          \

// dest = dest ∪ src
// treats both src and dest as sets
// [1,2,3] ∪ [2,4,4] = [1,2,3,4,4]
// TODO: if either src or dest are large and we're allowed to rearrange
// elements position then consider sorting
#define array_union(dest, src, cmp)                  \
  __extension__({                                    \
	uint32_t src_len = array_len((src));             \
	uint32_t dest_len = array_len((dest));           \
    for (uint i = 0; i < src_len; i++) {             \
      bool found = false;                            \
      for(uint j = 0; j < dest_len; j++) {           \
        if(cmp((dest)[j], (src)[i]) == 0) {          \
          found = true;                              \
          break;                                     \
        }                                            \
      }                                              \
      if(!found) {                                   \
        array_append((dest), (src)[i]);              \
      }                                              \
    }                                                \
})                                                   \

#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif
#endif
