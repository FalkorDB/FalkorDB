---
name: Verify index usage
description: Confirm queries are using indexes through execution plan analysis with GRAPH.EXPLAIN
---

# Verify index usage

Confirm that your queries are using indexes through execution plan analysis.

## Usage

Use `GRAPH.EXPLAIN` to verify that the query execution plan includes index scan operations.

## Example

    redis-cli GRAPH.EXPLAIN social "MATCH (p:Person) WHERE p.age = 30 RETURN p"

## Notes

- Look for "Index Scan" operations in the execution plan
- If you see "Label Scan" or "All Node Scan", the index is not being used
- Indexes are only used when predicates match the indexed properties
- Ensure indexes exist before expecting them to be used
- Some query patterns may prevent index usage (e.g., not-equal operators)
