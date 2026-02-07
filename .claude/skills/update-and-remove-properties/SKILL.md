---
name: Update and remove properties safely
description: Update node properties using SET and remove properties by setting them to NULL in FalkorDB
---

# Update and remove properties safely

Update node properties using `SET` and remove properties by setting them to `NULL`.

## Usage

Use `SET` to update or add properties. To remove a property, set it to `NULL`. Note that FalkorDB does not support the `REMOVE` keyword.

## Example

    redis-cli GRAPH.QUERY social "MATCH (u:User {id: 42})
    SET u.email = 'dana@example.com', u.temp = NULL"

## Notes

- `SET` can update existing properties or add new ones
- Setting a property to `NULL` effectively removes it
- FalkorDB does not support the `REMOVE` keyword - use `NULL` instead
- Multiple property operations can be chained in a single `SET` clause
