import streamlit as st


def create_supabase_table_and_function(client, table_name="documents"):
    """Create documents table + match function for Supabase"""
    ddl = f"""
    create table if not exists {table_name} (
        id uuid primary key default gen_random_uuid(),
        content text,
        metadata jsonb,
        embedding vector(768)
    );
    """
    fn = f"""
    create or replace function match_{table_name}(
        query_embedding vector(768),
        match_count int DEFAULT 5
    ) returns table (
        id uuid,
        content text,
        metadata jsonb,
        similarity float
    )
    language sql stable as $$
        select
            id,
            content,
            metadata,
            1 - (embedding <=> query_embedding) as similarity
        from {table_name}
        order by embedding <=> query_embedding
        limit match_count;
    $$;
    """
    client.rpc("execute_sql", {"sql": ddl}).execute()
    client.rpc("execute_sql", {"sql": fn}).execute()
    st.success(f"✅ Supabase table + function ready: {table_name}, match_{table_name}")


def file_already_indexed_supabase(client, table_name, file_hash):
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
