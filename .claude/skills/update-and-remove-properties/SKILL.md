---
name: Update and remove properties safely
description: Update node properties using SET and REMOVE, or by setting them to NULL in FalkorDB
---

# Update and remove properties safely

Update node properties using `SET` and remove properties using `REMOVE` or by setting them to `NULL`.

## Usage

Use `SET` to update or add properties. To remove a property, use `REMOVE` or set it to `NULL`. To remove labels, use `REMOVE`.

## Example

    redis-cli GRAPH.QUERY social "MATCH (u:User {id: 42})
    SET u.email = 'dana@example.com' REMOVE u.temp"

    redis-cli GRAPH.QUERY social "MATCH (u:User {id: 42})
    SET u.email = 'dana@example.com', u.temp = NULL"

    redis-cli GRAPH.QUERY social "MATCH (n:Foo:Bar) REMOVE n:Bar RETURN n"

## Notes

- `SET` can update existing properties or add new ones
- `REMOVE` can remove properties (`REMOVE n.prop`) and labels (`REMOVE n:Label`)
- Setting a property to `NULL` also removes it (alternative to `REMOVE`)
- Multiple property operations can be chained in a single `SET` clause
