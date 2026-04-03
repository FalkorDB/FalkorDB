#pragma once
/* arr.h - simple, easy to use dynamic array with fat pointers,
 * to allow native access to members. It can accept pointers, struct literals and scalars.
 *
 * Example usage:
 *
 *  int *arr = arr_new(int, 8);
 *  // Add elements to the array
 *  for (int i = 0; i < 100; i++) {
 *   arr_append(arr, i);
 *  }
 *
 *  // read individual elements
 *  for (int i = 0; i < arr_len(arr); i++) {
 *    printf("%d\n", arr[i]);
 *  }
 *
 *  arr_free(arr);
 *
 *
 *  */
#include "RG.h"
#include "rmalloc.h"

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/param.h>

#ifdef __cplusplus
extern "C" {
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
} arr_hdr_t;

typedef void *arr_t;
/* Internal - calculate the array size for allocations */
#define arr_sizeof(hdr) (sizeof(arr_hdr_t) + (uint64_t)hdr->cap * hdr->elem_sz)
/* Internal - get a pointer to the array header */
#define arr_hdr(arr) ((arr_hdr_t *)(((char *)arr) - sizeof(arr_hdr_t)))

static inline uint32_t arr_len(arr_t arr);

/* Initialize a new array with a given element size and capacity. Should not be used directly - use
 * arr_new instead */
static arr_t arr_new_sz(uint32_t elem_sz, uint32_t cap, uint32_t len) {
	uint64_t arr_size = (uint64_t)cap * elem_sz;
	arr_hdr_t *hdr = (arr_hdr_t *)rm_malloc(sizeof(arr_hdr_t) + arr_size);
	hdr->cap = cap;
	hdr->elem_sz = elem_sz;
	hdr->len = len;
	return (arr_t)(hdr->buf);
}

/* Initialize an array for a given type T with a given capacity and zero length. The array should be
 * case to a pointer to that type. e.g.
 *
 *  int *arr = arr_new(int, 4);
 *
 * This allows direct access to elements
 *  */
#define arr_new(T, cap) (T *)(arr_new_sz(sizeof(T), cap, 0))

/* initialize an array for a given type T with a given length
 * the capacity allocated is identical
 * to the length
 *  */
#define arr_newlen(T, len) (T *)(arr_new_sz(sizeof(T), len, len))

static inline arr_t arr_ensure_cap(arr_t arr, uint32_t cap) {
	arr_hdr_t *hdr = arr_hdr(arr);
	if(cap > hdr->cap) {
		hdr->cap = MAX(hdr->cap * 2, cap);
		hdr = (arr_hdr_t *)rm_realloc(hdr, arr_sizeof(hdr));
	}
	return (arr_t)hdr->buf;
}

// Reallocates the array setting the new capacity. If there were more
// elements, those are deleted (cut off), if less, then there will be
// space enough to fit (capacity - length) elements.
static inline arr_t arr_reset_cap(arr_t arr, const uint32_t cap) {
  arr_hdr_t *hdr = arr_hdr(arr);
  hdr->cap = cap;
  hdr->len = hdr->len > hdr->cap ? hdr->cap : hdr->len;
  hdr = (arr_hdr_t *)rm_realloc(hdr, arr_sizeof(hdr));
	return (arr_t)hdr->buf;
}

/* Ensure capacity for the array to grow by one */
static inline arr_t arr_grow(arr_t arr, size_t n) {
	arr_hdr(arr)->len += n;
	return arr_ensure_cap(arr, arr_hdr(arr)->len);
}

static inline arr_t arr_ensure_len(arr_t arr, size_t len) {
	if(len <= arr_len(arr)) {
		return arr;
	}
	len -= arr_len(arr);
	return arr_grow(arr, len);
}

/* Ensures that arr_tail will always point to a valid element. */
#define arr_ensure_tail(arrpp, T)              \
  ({                                           \
    if (!*(arrpp)) {                           \
      *(arrpp) = arr_newlen(T, 1);             \
    } else {                                   \
      *(arrpp) = (T *)arr_grow(*(arrpp), 1);   \
    }                                          \
    &(arr_tail(*(arrpp)));                     \
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
#define arr_ensure_append(arrpp, src, n, T) do {      \
    size_t a__oldlen = 0;                             \
    if (!arrpp) {                                     \
      (arrpp) = arr_newlen(T, n);                     \
    } else {                                          \
      a__oldlen = arr_len(arrpp);                     \
      (arrpp) = (T *)arr_grow(arrpp, n);              \
    }                                                 \
    memcpy((arrpp) + a__oldlen, src, n * sizeof(T));  \
  } while(0)

/**
 * Does the same thing as ensure_append, but the added elements are
 * at the _beginning_ of the array
 */
#define arr_ensure_prepend(arrpp, src, n, T)                            \
  ({                                                                    \
    size_t a__oldlen = 0;                                               \
    if (!arrpp) {                                                       \
      arrpp = arr_newlen(T, n);                                         \
    } else {                                                            \
      a__oldlen = arr_len(arrpp);                                       \
      arrpp = (T *)arr_grow(arrpp, n);                                  \
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
#define arr_ensure_at(arrpp, pos, T)                                      \
  ({                                                                      \
    if (!(*arrpp)) {                                                      \
      *(arrpp) = arr_new(T, 1);                                           \
    }                                                                     \
    if (arr_len(*arrpp) <= pos) {                                         \
      size_t curlen = arr_len(*arrpp);                                    \
      arr_hdr(*arrpp)->len = pos + 1;                                     \
      *arrpp = (T *)arr_ensure_cap(*(arrpp), arr_hdr(*(arrpp))->len);     \
      memset((T *)*arrpp + curlen, 0, sizeof(T) * ((pos + 1) - curlen));  \
    }                                                                     \
    (T *)(*arrpp) + pos;                                                  \
  })

/* get the last element in the array */
#define arr_tail(arr) ((arr)[arr_hdr(arr)->len - 1])

/* Append an element to the array, returning the array which may have been reallocated */
#define arr_append(arr, x) do {                      \
    (arr) = (__typeof__(arr))arr_grow((arr), 1);     \
    arr_tail((arr)) = (x);                           \
  } while(0)

/* Get the length of the array */
static inline uint32_t arr_len(const arr_t arr) {
	return arr ? arr_hdr(arr)->len : 0;
}

// get array's cap
static inline uint32_t arr_cap
(
	const arr_t arr
) {
	return arr_hdr(arr)->cap ;
}

#define ARR_CAP_NOSHRINK ((uint32_t)-1)
static inline void *arr_trimm(arr_t arr, uint32_t len, uint32_t cap) {
	arr_hdr_t *arr_hdr = arr_hdr(arr);
	ASSERT((cap == ARR_CAP_NOSHRINK || cap > 0 || len == cap) && "trimming capacity is illegal");
	ASSERT((cap == ARR_CAP_NOSHRINK || cap >= len) && "trimming len is greater then capacity");
	ASSERT((len <= arr_hdr->len) && "trimming len is greater then current len");
	arr_hdr->len = len;
	if(cap != ARR_CAP_NOSHRINK) {
		arr_hdr->cap = cap;
		arr_hdr = (arr_hdr_t *)rm_realloc(arr_hdr, arr_sizeof(arr_hdr));
	}
	return arr_hdr->buf;
}

#define arr_trimm_len(arr, len) (__typeof__(arr)) arr_trimm(arr, len, ARR_CAP_NOSHRINK)
#define arr_trimm_cap(arr, len) (__typeof__(arr)) arr_trimm(arr, len, len)

/* Free the array, without dealing with individual elements */
static void arr_free(arr_t arr) {
	if(arr != NULL) {
		// like free(), shouldn't explode if NULL
		rm_free(arr_hdr(arr));
	}
}

#define arr_clear(arr) arr_hdr(arr)->len = 0

/* Free the array, free individual element using callback */
#define arr_free_cb(arr, cb)                          \
  ({                                                  \
    if (arr) {                                        \
      for (uint32_t i = 0; i < arr_len(arr); i++) {   \
        { cb(arr[i]); }                               \
      }                                               \
      arr_free(arr);                                  \
    }                                                 \
  })

/* Pop the top element from the array, reduce the size and return it */
#define arr_pop(arr)                 \
  __extension__ ({                   \
    ASSERT(arr_hdr(arr)->len > 0);   \
    (arr)[--(arr_hdr(arr)->len)];    \
  })

/* Remove a specified element from the array */
#define arr_del(arr, ix)                                                           \
  __extension__({                                                                  \
    ASSERT(arr_len(arr) > ix);                                                     \
    if (arr_len(arr) - 1 > ix) {                                                   \
      memmove(arr + ix, arr + ix + 1, sizeof(*arr) * (arr_len(arr) - (ix + 1)));   \
    }                                                                              \
    --arr_hdr(arr)->len;                                                           \
    arr;                                                                           \
  })

/* Remove a specified element from the array, but does not preserve order */
#define arr_del_fast(arr, ix)                  \
  __extension__({                              \
    if (arr_len((arr)) > 1) {                  \
      (arr)[ix] = (arr)[arr_len((arr)) - 1];   \
    }                                          \
    --arr_hdr((arr))->len;                     \
    arr;                                       \
  })

/* Duplicate the array to the pointer dest. */
#define arr_clone(dest, arr)                              \
  __extension__({                                         \
   dest = arr_newlen(typeof(*arr), arr_len(arr));         \
   memcpy(dest, arr, sizeof(*arr) * (arr_len(arr)));      \
  })

/* Duplicate an array with a dedicated value clone callback. */
#define arr_clone_with_cb(dest, arr, clone_cb)          \
__extension__({                                         \
    uint arrayLen = arr_len((arr));                     \
    dest = arr_new(__typeof__(*arr), arrayLen);         \
    for(uint i = 0; i < arrayLen; i++)                  \
        arr_append(dest, (clone_cb(arr[i])));           \
})

#define arr_reverse(arr)                        \
    __extension__({                             \
        uint arrayLen = arr_len(arr);           \
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
#define arr_union(dest, src, cmp)                    \
  __extension__({                                    \
	uint32_t src_len = arr_len((src));               \
	uint32_t dest_len = arr_len((dest));             \
    for (uint i = 0; i < src_len; i++) {             \
      bool found = false;                            \
      for(uint j = 0; j < dest_len; j++) {           \
        if(cmp((dest)[j], (src)[i]) == 0) {          \
          found = true;                              \
          break;                                     \
        }                                            \
      }                                              \
      if(!found) {                                   \
        arr_append((dest), (src)[i]);                \
      }                                              \
    }                                                \
})                                                   \

#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif

