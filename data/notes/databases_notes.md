# Databases Notes

## ACID Properties
Transactions in relational databases follow ACID:
- **Atomicity**: a transaction either fully completes or fully fails, no partial state.
- **Consistency**: a transaction brings the database from one valid state to another.
- **Isolation**: concurrent transactions don't interfere with each other's intermediate state.
- **Durability**: once committed, changes survive crashes/power loss.

## Normalization
- 1NF: atomic column values, no repeating groups.
- 2NF: 1NF + no partial dependency on a composite key.
- 3NF: 2NF + no transitive dependency on non-key attributes.

## Indexes
B-tree indexes speed up lookups but slow down writes since the index must
also be updated on every insert/update/delete.
