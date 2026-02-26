---
name: Manage constraints with awareness of async creation
description: Create and check status of constraints on node properties in FalkorDB with db.constraints()
---

# Manage constraints with awareness of async creation

Create and manage constraints on node properties, understanding that constraint creation may be asynchronous.

## Usage

Use `GRAPH.CONSTRAINT CREATE` to establish constraints and check their status with the `db.constraints()` procedure.

## Example

    redis-cli GRAPH.QUERY social "CREATE INDEX FOR (p:Person) ON (p.id)"
    redis-cli GRAPH.CONSTRAINT CREATE social UNIQUE NODE Person PROPERTIES 1 id
    redis-cli GRAPH.QUERY social "CALL db.constraints()"

## Notes

- A supporting exact-match index must exist on the property before creating a UNIQUE constraint
- Constraint creation is asynchronous; it returns `PENDING` and builds in the background
- Use `db.constraints()` to verify constraint status (check for `OPERATIONAL` status)
- Constraints help prevent duplicate data and maintain data quality
- Constraint violations will cause write operations to fail
