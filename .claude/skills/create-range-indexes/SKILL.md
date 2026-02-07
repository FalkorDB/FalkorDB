---
name: Create range indexes for exact/range lookups
description: Create indexes to accelerate equality and range predicates on node properties in FalkorDB
---

# Create range indexes for exact/range lookups

Create indexes to accelerate equality and range predicates on node properties.

## Usage

Use `CREATE INDEX FOR` to create an index on a node label and property combination.

## Example

    redis-cli GRAPH.QUERY social "CREATE INDEX FOR (p:Person) ON (p.age)"

## Notes

- Range indexes speed up both exact matches and range queries
- Indexes improve performance for `WHERE` clauses with equality or range predicates
- Index creation is synchronous and may take time for large datasets
- Multiple indexes can be created on different properties
- Indexes consume additional memory but significantly improve query performance
