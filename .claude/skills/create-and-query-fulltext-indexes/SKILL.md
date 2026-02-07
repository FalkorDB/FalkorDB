---
name: Create and query full-text indexes
description: Use RediSearch-backed full-text indexes for text search with wildcard and fuzzy matching
---

# Create and query full-text indexes

Use RediSearch-backed full-text indexes for text search capabilities.

## Usage

Create full-text indexes using the `db.idx.fulltext.createNodeIndex` procedure, then query them using `db.idx.fulltext.queryNodes`.

## Example

    redis-cli GRAPH.QUERY social "CALL db.idx.fulltext.createNodeIndex('Movie', 'title')"
    redis-cli GRAPH.QUERY social "CALL db.idx.fulltext.queryNodes('Movie', 'Jun*') YIELD node RETURN node.title"

## Notes

- Full-text indexes leverage RediSearch for advanced text search capabilities
- Supports wildcard searches and fuzzy matching
- Use `CALL` to invoke the index creation and query procedures
- `YIELD node` is used to capture results from the procedure
- Full-text indexes are ideal for searching text content with partial matches
