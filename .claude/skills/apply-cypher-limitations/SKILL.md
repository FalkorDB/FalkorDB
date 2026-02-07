---
name: Apply FalkorDB Cypher limitations correctly
description: Account for FalkorDB Cypher limitations like non-indexed not-equal filters when designing queries
---

# Apply FalkorDB Cypher limitations correctly

Account for known limitations in FalkorDB's Cypher implementation when designing queries.

## Usage

Be aware of FalkorDB-specific limitations and design queries accordingly to avoid performance issues.

## Example

    redis-cli GRAPH.QUERY social "MATCH (p:Person) WHERE p.age <> 30 RETURN p"

## Notes

- Not-equal (`<>` or `!=`) filters cannot use indexes and will perform full scans
- Certain Cypher features may have different behavior in FalkorDB compared to other graph databases
- Always test queries with representative data to understand performance characteristics
- Review FalkorDB documentation for the complete list of limitations
- Plan query design around these limitations for optimal performance
