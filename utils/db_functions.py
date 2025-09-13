def file_already_indexed_supabase(client, table_name, file_hash):
    """Check if at least one chunk for this file exists"""
    try:
        res = (
            client.table(table_name)
            .select("id")
            .eq("metadata->>file_hash", file_hash)
            .limit(1)
            .execute()
        )
        return len(res.data) > 0
    except Exception:
        return False


def get_existing_chunks_for_file(client, table_name, file_hash):
    """Return a set of chunk_index values already stored for this file"""
    try:
        res = (
            client.table(table_name)
            .select("metadata")
            .eq("metadata->>file_hash", file_hash)
            .execute()
        )
        return {
            row["metadata"].get("chunk_index")
            for row in res.data
            if row.get("metadata")
        }
    except Exception:
        return set()
