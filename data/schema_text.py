from __future__ import annotations

from typing import Any


def format_table_block(
    table_name: str,
    columns: list[dict[str, Any]],
    foreign_keys: list[dict[str, str]] | None = None,
) -> str:
    """
    Shared compact schema block used by monolithic baseline

    This guarantees identical table formatting.
    """
    foreign_keys = foreign_keys or []

    lines = [f"Table: {table_name}"]

    for col in columns:
        pk_mark = " [PK]" if col.get("pk", False) else ""
        lines.append(f"- {col['name']} ({col['type']}){pk_mark}")

    if foreign_keys:
        lines.append("")
        lines.append("Foreign keys:")
        for fk in foreign_keys:
            lines.append(f"- {fk['from']} -> {fk['to']}")

    return "\n".join(lines)


def format_full_schema_from_json(schema_json: dict[str, Any]) -> str:
    """
    Render the full schema in the same compact table-block format
    used by retrieve_from_list.
    """
    tables = schema_json.get("tables", [])
    all_fks = schema_json.get("foreign_keys", [])

    fk_map: dict[str, list[dict[str, str]]] = {}
    for fk in all_fks:
        source_table = fk["source_table"]
        target_table = fk["target_table"]

        fk_entry = {
            "from": f"{source_table}.{fk['source_column']}",
            "to": f"{target_table}.{fk['target_column']}",
        }

        fk_map.setdefault(source_table, []).append(fk_entry)
        fk_map.setdefault(target_table, []).append(fk_entry)

    blocks = []
    for table in tables:
        table_name = table["table_name"]
        cols = []
        for col in table.get("columns", []):
            cols.append({
                "name": col["column_name"],
                "type": col["column_type"],
                "pk": bool(col.get("is_primary_key", False)),
            })

        table_fks = fk_map.get(table_name, [])
        blocks.append(format_table_block(table_name, cols, table_fks))

    return "\n\n".join(blocks)


def format_table_block_from_database_schema(schema, table_name: str) -> str:
    """
    Render one table block from the internal Spider DB schema object
    using the SAME compact format as the monolithic baseline.
    """
    table_idx = schema.table_names_original.index(table_name)

    columns = []
    for col_idx, (t_idx, col_name) in enumerate(schema.column_names_original):
        if t_idx != table_idx:
            continue

        col_type = (
            schema.column_types[col_idx]
            if col_idx < len(schema.column_types)
            else "unknown"
        )

        columns.append({
            "name": col_name,
            "type": col_type,
            "pk": col_idx in schema.primary_keys,
        })

    foreign_keys = []
    for source_col_idx, target_col_idx in schema.foreign_keys:
        source_table_idx, source_col_name = schema.column_names_original[source_col_idx]
        target_table_idx, target_col_name = schema.column_names_original[target_col_idx]

        source_table = (
            schema.table_names_original[source_table_idx]
            if source_table_idx >= 0 else None
        )
        target_table = (
            schema.table_names_original[target_table_idx]
            if target_table_idx >= 0 else None
        )

        if source_table == table_name or target_table == table_name:
            foreign_keys.append({
                "from": f"{source_table}.{source_col_name}",
                "to": f"{target_table}.{target_col_name}",
            })

    return format_table_block(table_name, columns, foreign_keys)