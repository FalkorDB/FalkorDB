---
name: Manage constraints with awareness of async creation
description: Create and check status of constraints on node properties in FalkorDB with db.constraints()
---

# Manage constraints with awareness of async creation

Create and manage constraints on node properties, understanding that constraint creation may be asynchronous.

## Usage

Use `GRAPH.CONSTRAINT CREATE` to establish constraints and check their status with the `db.constraints()` procedure.

## Example

    redis-cli GRAPH.CONSTRAINT CREATE social UNIQUE NODE Person PROPERTIES 1 id
    redis-cli GRAPH.QUERY social "CALL db.constraints()"

## Notes

- Constraints ensure data integrity by enforcing uniqueness rules
- Constraint creation may be asynchronous, especially on large graphs
- Use `db.constraints()` to verify constraint status
- The syntax shown is typical, but confirm exact syntax with current FalkorDB documentation
- Constraints help prevent duplicate data and maintain data quality
- Constraint violations will cause write operations to fail
