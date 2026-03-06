---
name: Run safe read-only queries
description: Use GRAPH.RO_QUERY for read-only operations that reject write attempts in FalkorDB
---

# Run safe read-only queries

Use `GRAPH.RO_QUERY` for read-only operations that reject any write attempts.

## Usage

Replace `GRAPH.QUERY` with `GRAPH.RO_QUERY` for queries that should never modify the graph.

## Example

    redis-cli GRAPH.RO_QUERY social "MATCH (u:User) RETURN count(u)"

## Notes

- `GRAPH.RO_QUERY` enforces read-only semantics at the API level
- Any attempt to write data (CREATE, SET, DELETE, MERGE) will be rejected
- Useful for ensuring query safety in production environments
- Can help prevent accidental data modifications
- Provides an additional layer of data protection
