/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "datablock.h"
#include "datablock_iterator.h"
#include "../arr.h"
#include "../rmalloc.h"
#include <math.h>
#include <stdbool.h>

// computes the number of blocks required to accommodate n items.
#define ITEM_COUNT_TO_BLOCK_COUNT(n, cap) \
    ceil((double)n / cap)

// computes block index from item index.
#define ITEM_INDEX_TO_BLOCK_INDEX(idx, cap) \
    (idx / cap)

// computes item position within a block.
#define ITEM_POSITION_WITHIN_BLOCK(idx, cap) \
    (idx % cap)

// retrieves block in which item with index resides.
#define GET_ITEM_BLOCK(dataBlock, idx) \
    dataBlock->blocks[ITEM_INDEX_TO_BLOCK_INDEX(idx, dataBlock->blockCap)]

static void _DataBlock_AddBlocks
(
	DataBlock *dataBlock,
	uint blockCount
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (blockCount > 0) ;

	uint prevBlockCount = dataBlock->blockCount ;
	dataBlock->blockCount += blockCount ;
	if (!dataBlock->blocks) {
		dataBlock->blocks =
			rm_malloc (sizeof (Block *) * dataBlock->blockCount) ;
	} else {
		dataBlock->blocks =
			rm_realloc (dataBlock->blocks, sizeof (Block *) * dataBlock->blockCount) ;
	}

	uint i;
	for(i = prevBlockCount; i < dataBlock->blockCount; i++) {
		dataBlock->blocks[i] = Block_New(dataBlock->itemSize, dataBlock->blockCap);
		if(i > 0) dataBlock->blocks[i - 1]->next = dataBlock->blocks[i];
	}
	dataBlock->blocks[i - 1]->next = NULL;

	dataBlock->itemCap = dataBlock->blockCount * dataBlock->blockCap;
}

// Checks to see if idx is within global array bounds
// array bounds are between 0 and itemCount + #deleted indices
// e.g. [3, 7, 2, D, 1, D, 5] where itemCount = 5 and #deleted indices is 2
// and so it is valid to query the array with idx 6.
static inline bool _DataBlock_IndexOutOfBounds
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	return (idx >= (dataBlock->itemCount + arr_len(dataBlock->deletedIdx)));
}

DataBlockItemHeader *DataBlock_GetItemHeader
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	Block *block = GET_ITEM_BLOCK(dataBlock, idx);
	idx = ITEM_POSITION_WITHIN_BLOCK(idx, dataBlock->blockCap);
	return (DataBlockItemHeader *)block->data + (idx * block->itemSize);
}

//------------------------------------------------------------------------------
// DataBlock API implementation
//------------------------------------------------------------------------------

DataBlock *DataBlock_New
(
	uint64_t blockCap,
	uint64_t itemCap,
	uint itemSize,
	fpDestructor fp
) {
	DataBlock *dataBlock = rm_malloc(sizeof(DataBlock));
	dataBlock->blocks     = NULL;
	dataBlock->itemSize   = itemSize + ITEM_HEADER_SIZE;
	dataBlock->itemCount  = 0;
	dataBlock->blockCount = 0;
	dataBlock->blockCap   = blockCap;
	dataBlock->deletedIdx = arr_new(uint64_t, 128);
	dataBlock->destructor = fp;

	_DataBlock_AddBlocks(dataBlock,
			ITEM_COUNT_TO_BLOCK_COUNT(itemCap, dataBlock->blockCap));

	return dataBlock;
}

uint64_t DataBlock_ItemCount(const DataBlock *dataBlock) {
	return dataBlock->itemCount;
}

// returns datablock item size
uint DataBlock_itemSize
(
	const DataBlock *dataBlock  // datablock
) {
	ASSERT(dataBlock != NULL);

	return dataBlock->itemSize;
}

DataBlockIterator *DataBlock_Scan(const DataBlock *dataBlock) {
	ASSERT(dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0];

	// Deleted items are skipped, we're about to perform
	// array_len(dataBlock->deletedIdx) skips during out scan.
	int64_t endPos = dataBlock->itemCount + arr_len(dataBlock->deletedIdx);
	return DataBlockIterator_New(startBlock, dataBlock->blockCap, endPos);
}

DataBlockIterator *DataBlock_FullScan(const DataBlock *dataBlock) {
	ASSERT(dataBlock != NULL);
	Block *startBlock = dataBlock->blocks[0];

	int64_t endPos = dataBlock->blockCount * dataBlock->blockCap;
	return DataBlockIterator_New(startBlock, dataBlock->blockCap, endPos);
}

// Make sure datablock can accommodate at least k items.
void DataBlock_Accommodate
(
	DataBlock *dataBlock,
	int64_t k
) {
	// compute number of free slots
	int64_t freeSlotsCount  = dataBlock->itemCap - dataBlock->itemCount ;
	int64_t additionalItems = k - freeSlotsCount ;

	if (additionalItems > 0) {
		int64_t additionalBlocks =
			ITEM_COUNT_TO_BLOCK_COUNT (additionalItems, dataBlock->blockCap) ;
		_DataBlock_AddBlocks (dataBlock, additionalBlocks) ;
	}
}

void DataBlock_Ensure(DataBlock *dataBlock, uint64_t idx) {
	ASSERT(dataBlock != NULL);

	// datablock[idx] exists
	if(dataBlock->itemCap > idx) return;

	// make sure datablock cap > 'idx'
	int64_t additionalItems = (1 + idx) - dataBlock->itemCap;
	int64_t additionalBlocks =
		ITEM_COUNT_TO_BLOCK_COUNT(additionalItems, dataBlock->blockCap);
	_DataBlock_AddBlocks(dataBlock, additionalBlocks);

	ASSERT(dataBlock->itemCap > idx);
}

void *DataBlock_GetItem
(
	const DataBlock *dataBlock,
	uint64_t idx
) {
	ASSERT (dataBlock != NULL) ;

	// return NULL if idx is out of bounds
	if (_DataBlock_IndexOutOfBounds (dataBlock, idx))
	{
		return NULL ;
	}

	DataBlockItemHeader *item_header = DataBlock_GetItemHeader (dataBlock, idx) ;

	// incase item is marked as deleted, return NULL.
	if (IS_ITEM_DELETED (item_header))
	{
		return NULL ;
	}

	return ITEM_DATA (item_header) ;
}

uint64_t DataBlock_GetReservedIdx
(
	const DataBlock *dataBlock,
	uint64_t n
) {
	ASSERT (dataBlock != NULL) ;

	uint deleted = DataBlock_DeletedItemsCount (dataBlock) ;
	if (n < deleted) {
		return dataBlock->deletedIdx[deleted - n - 1] ;
	} 

	return DataBlock_ItemCount (dataBlock) + n ;
}

void *DataBlock_AllocateItem
(
	DataBlock *dataBlock,
	uint64_t *idx
) {
	// make sure we've got room for items
	if (unlikely (dataBlock->itemCount >= dataBlock->itemCap)) {
		// allocate an additional block
		_DataBlock_AddBlocks (dataBlock, 1) ;
	}

	ASSERT (dataBlock->itemCap > dataBlock->itemCount) ;

	// get index into which to store item,
	// prefer reusing free indicies
	uint pos = dataBlock->itemCount ;
	if (arr_len (dataBlock->deletedIdx) > 0) {
		pos = arr_pop (dataBlock->deletedIdx) ;

		// trim array if number of free entries is greater than 20%
		if (unlikely (
			(float)arr_len (dataBlock->deletedIdx) /
			(float)arr_cap (dataBlock->deletedIdx) <= 0.8)
		) {
			dataBlock->deletedIdx = arr_trimm_cap (dataBlock->deletedIdx,
					arr_len (dataBlock->deletedIdx)) ;
		}
	}

	dataBlock->itemCount++ ;

	if (idx) {
		*idx = pos;
	}

	DataBlockItemHeader *header = DataBlock_GetItemHeader (dataBlock, pos) ;
	MARK_HEADER_AS_NOT_DELETED (header) ;

	return ITEM_DATA (header) ;
}

void DataBlock_DeleteItem(DataBlock *dataBlock, uint64_t idx) {
	ASSERT(dataBlock != NULL);
	ASSERT(!_DataBlock_IndexOutOfBounds(dataBlock, idx));

	// Return if item already deleted.
	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);
	if(IS_ITEM_DELETED(item_header)) return;

	// Call item destructor.
	if(dataBlock->destructor) {
		unsigned char *item = ITEM_DATA(item_header);
		dataBlock->destructor(item);
	}

	MARK_HEADER_AS_DELETED(item_header);

	arr_append(dataBlock->deletedIdx, idx);
	dataBlock->itemCount--;
}

uint DataBlock_DeletedItemsCount(const DataBlock *dataBlock) {
	return arr_len(dataBlock->deletedIdx);
}

inline bool DataBlock_ItemIsDeleted(void *item) {
	DataBlockItemHeader *header = GET_ITEM_HEADER(item);
	return IS_ITEM_DELETED(header);
}

// returns datablock's deleted indices array
const uint64_t *DataBlock_DeletedItems
(
	const DataBlock *dataBlock
) {
	ASSERT(dataBlock != NULL);

	return (const uint64_t *) dataBlock->deletedIdx;
}

//------------------------------------------------------------------------------
// Out of order functionality
//------------------------------------------------------------------------------

// claim a BATCH of specific ids for live allocation
//
// This exists because DataBlock_AllocateItemOutOfOrder below carries a
// precondition the effects path cannot meet.
//
// That one marks the header and bumps itemCount and never touches 'deletedIdx',
// so it REQUIRES that 'idx' is not on the free list. Its callers are the RDB
// decoders, where that holds by construction: the file stores live entities and
// deleted ids as two separate lists, the deleted list is restored through
// DataBlock_MarkAsDeletedOutOfOrder (which does append), and the live list
// through AllocateItemOutOfOrder. The two id sets are disjoint, so there is
// never anything to remove. It is correct there and stays in use.
//
// Effects apply violates that precondition head-on: the ordinary case for a
// replica is a create naming an id that IS on the free list, because that is
// what reuse means. Used there, the id would stay on the list and the next
// allocation would hand it out a second time, putting two entities on one id.
//
// Used by effects apply, where the id is not this node's to choose. C's
// allocator reuses the most recently freed id and Rust's reuses the smallest,
// so after any delete-then-create cycle the two disagree about which id a new
// entity gets, and the replica must take the id the primary states.
//
// A BATCH RATHER THAN ONE AT A TIME, and that is the whole point of the
// signature. Claiming ids one by one means finding each on the free list,
// which is a flat array, so it is a scan per id. Claiming ascending ids out of
// an ascending list finds entry j at roughly position j, and the positions sum
// rather than cancel: measured at 0.7 ms for 1,000 ids and 237.8 ms for
// 64,000, growing 3.89x per doubling against the 2x of linear, with the scan
// 93% of the record at the top end. This path had already produced two
// quadratics - the per-edge GB_wait and the UPDATE_EDGE inner scan - and a
// third was not worth shipping.
//
// So the free list is walked ONCE per call. It is not searched at all: the
// claimed ids are marked live first, which makes the HEADER the membership
// test, and one pass keeps whatever is still marked deleted. O(k) for a free
// list of k, against the O(n*k) of claiming one at a time.
//
// A CURSOR RATHER THAN AN ARRAY, because the batch size comes off the wire.
// One Range segment describes four billion ids in about seven bytes, so an
// array indexed by the stated count is an allocation the payload never pays
// for - 28 bytes of record drove 2,229 MB of resident memory on an earlier
// build. The ids are walked twice and held once, by whatever decoded them.
//
// Three cases, and the caller only needs to distinguish the last:
//
//   the id is on the free list   claim it, and drop it from the list
//   the id is past high water    extend, and every id SKIPPED OVER becomes
//                                free rather than lost - a leaked id makes
//                                this replica's own allocation diverge the
//                                moment it allocates anything itself, which
//                                is invisible until promotion
//   the id is live               genuine divergence
//
// NOTHING IS CLAIMED when this returns false - and no capacity is added
// either, so a refusal leaves this block byte for byte as it was.
//
// returns false if any id is already live, if the walk is not strictly
// ascending, if it disagrees with the count it stated, or if an id is past the
// highest this batch could legitimately reach
bool DataBlock_ClaimIds
(
	DataBlock *dataBlock,
	IdCursor *ids  // ascending walk over the ids to claim
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (ids       != NULL) ;

	const uint64_t n = ids->count ;

	if (n == 0) {
		return true ;
	}

	// every id below this is either live or on the free list; every id at or
	// above it has never been handed out. Taken BEFORE anything is changed.
	const uint64_t high_water =
		dataBlock->itemCount + (uint64_t)arr_len (dataBlock->deletedIdx) ;

	// NO CEILING ON THE ID, deliberately, and this is the second time that
	// lesson has been paid for. A record's ids are NOT dense in the id space
	// and cannot be made to be: v3 emits one record per (opcode, shape) and
	// sorts the groups by key, so one query's ids are split across records and
	// every record's list is sparse. CREATE (:A), (:B), (:A) allocates 0,1,2
	// and sends TWO records - ':A' holding {0, 2} and ':B' holding {1} - so a
	// bound of "high water plus this record's count" refuses the first of them
	// on an empty graph. Rust shipped two such ceilings and both fired on
	// legitimate data, each putting a replica into a resync loop that failed
	// again identically.
	//
	// UINT64_MAX is still refused, for a reason that is not a guess about the
	// peer: it is INVALID_ENTITY_ID, so no entity can carry it, and it is
	// where DataBlock_Ensure's (1 + idx) wraps to zero and the write lands off
	// the end of the block.
	//
	// SO AN ARBITRARY ID STILL SIZES AN ALLOCATION HERE. A record naming one
	// entity at id 2^40 grows this block to 2^40 items - measured at 1,155 MB
	// for a single node at id 67,108,864. Nothing derived from the payload
	// bounds that, because a sparse record is what correct grouping produces
	// and the slots it skips are the slots the primary also has. It is a
	// resource limit to be decided, not a validation that can be written here.

	//--------------------------------------------------------------------------
	// pass 1: decide, changing nothing
	//--------------------------------------------------------------------------
	//
	// Not even capacity. Every id below the high-water mark already has a
	// slot, and no id above it is touched here, so there is nothing to
	// Ensure yet.
	uint64_t seen = 0 ;
	uint64_t prev = 0 ;
	uint64_t id ;

	ids->reset (ids) ;

	while (ids->next (ids, &id)) {
		// a walk that outruns the count it stated is caught per step rather
		// than after it, so no cursor can make this loop unbounded
		if (seen == n || id == UINT64_MAX) {
			return false ;
		}

		// STRICTLY ascending, which IS the duplicate check and the only place
		// it happens. A cursor promises non-decreasing and no more - a Repeat
		// segment names one id several times - so this is not a second opinion
		// on something already guaranteed. In order a repeated id is the one
		// before it, and a create must refuse it: two entities cannot share a
		// slot.
		if (seen > 0 && id <= prev) {
			return false ;
		}

		// an id already live is divergence. Asked only BELOW the high-water
		// mark: a block is rm_calloc'd, so a slot that has never been handed
		// out reads as NOT deleted and would answer this question wrongly.
		if (id < high_water &&
			!IS_ITEM_DELETED (DataBlock_GetItemHeader (dataBlock, id))) {
			return false ;
		}

		prev = id ;
		seen++ ;
	}

	if (seen != n) {
		// the walk ran short of the count it stated
		return false ;
	}

	//--------------------------------------------------------------------------
	// pass 2: claim
	//--------------------------------------------------------------------------
	//
	// ONE Ensure rather than one per id: the walk is ascending, so the id it
	// ended on is the highest, and every slot below it is covered by the same
	// call. This is where an arbitrary id becomes an arbitrary allocation -
	// see the note above the first pass.
	DataBlock_Ensure (dataBlock, prev) ;

	uint64_t next    = high_water ;  // first slot never handed out
	uint64_t claimed = 0 ;

	ids->reset (ids) ;

	while (claimed < n && ids->next (ids, &id)) {
		// every id SKIPPED OVER past the high-water mark becomes free rather
		// than lost. In practice this runs zero times, because a primary's ids
		// are dense - it is here for when they are not.
		if (id >= next) {
			for (uint64_t j = next ; j < id ; j++) {
				DataBlockItemHeader *skipped =
					DataBlock_GetItemHeader (dataBlock, j) ;
				MARK_HEADER_AS_DELETED (skipped) ;
				arr_append (dataBlock->deletedIdx, j) ;
			}
			next = id + 1 ;
		}

		MARK_HEADER_AS_NOT_DELETED (DataBlock_GetItemHeader (dataBlock, id)) ;
		claimed++ ;
	}

	// RELEASE_ASSERT, not ASSERT: the two walks are the same walk, and a
	// cursor that disagreed with itself would have claimed a different set
	// from the one pass 1 cleared. Carrying on means itemCount counts slots
	// that were never marked live while the caller writes entities into them
	// through DataBlock_ClaimedItem, whose own bounds check is an ASSERT and
	// so is not there in release either. Stopping is the lesser outcome, and
	// the condition is a bug in a cursor rather than anything off the wire.
	RELEASE_ASSERT (claimed == n) ;

	//--------------------------------------------------------------------------
	// drop the claimed ids from the free list, in ONE pass
	//--------------------------------------------------------------------------
	//
	// THE HEADER IS THE MEMBERSHIP TEST. The claimed ids were marked live just
	// above, so an entry still marked deleted is one this batch did not take -
	// no sorted copy of the ids, and no search per entry. The gap slots
	// appended just above are marked deleted, so they are kept, which is what
	// they are for.
	//
	// The list is a SET stored in an array and popped from the back, so its
	// order carries no meaning and compacting in place preserves it. A replica
	// never allocates while it is a replica, so the order is only observable
	// after promotion, and then only as which free id is reused first.
	const uint32_t k = arr_len (dataBlock->deletedIdx) ;
	uint32_t w = 0 ;

	for (uint32_t i = 0 ; i < k ; i++) {
		const uint64_t free_id = dataBlock->deletedIdx[i] ;
		if (IS_ITEM_DELETED (DataBlock_GetItemHeader (dataBlock, free_id))) {
			dataBlock->deletedIdx[w++] = free_id ;
		}
	}

	for (uint32_t i = w ; i < k ; i++) {
		arr_pop (dataBlock->deletedIdx) ;
	}

	dataBlock->itemCount += claimed ;

	return true ;
}

// the storage slot of an id this DataBlock has already claimed
void *DataBlock_ClaimedItem
(
	const DataBlock *dataBlock,
	uint64_t id
) {
	ASSERT (dataBlock != NULL) ;
	ASSERT (id < dataBlock->itemCap) ;

	return ITEM_DATA (DataBlock_GetItemHeader (dataBlock, id)) ;
}

void *DataBlock_AllocateItemOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// Check if idx<=data block's current capacity. If needed, allocate additional blocks.
	DataBlock_Ensure(dataBlock, idx);
	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);
	MARK_HEADER_AS_NOT_DELETED(item_header);
	dataBlock->itemCount++;
	return ITEM_DATA(item_header);
}

void DataBlock_MarkAsDeletedOutOfOrder
(
	DataBlock *dataBlock,
	uint64_t idx
) {
	// check if idx<=data block's current capacity
	// if needed, allocate additional blocks
	DataBlock_Ensure(dataBlock, idx);
	DataBlockItemHeader *item_header = DataBlock_GetItemHeader(dataBlock, idx);

	// delete
	MARK_HEADER_AS_DELETED(item_header);
	arr_append(dataBlock->deletedIdx, idx);
}

size_t DataBlock_memoryUsage
(
	const DataBlock *dataBlock
) {
	ASSERT(dataBlock != NULL);

	// datablock size = deleted index array size +
	//                  (number of blocks * block size)
	size_t data_size = sizeof(DataBlock) ;
	data_size += arr_bytesize(dataBlock->deletedIdx);
	data_size += RedisModule_MallocSize(dataBlock->blocks) ;
	data_size += dataBlock->blockCount
		* (sizeof (Block) + dataBlock->itemSize * dataBlock->blockCap);
	return data_size;
}

void DataBlock_Free(DataBlock *dataBlock) {
	for(uint i = 0; i < dataBlock->blockCount; i++) Block_Free(dataBlock->blocks[i]);

	rm_free(dataBlock->blocks);
	arr_free(dataBlock->deletedIdx);
	rm_free(dataBlock);
}

