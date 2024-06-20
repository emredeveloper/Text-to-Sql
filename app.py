import os
import hashlib
import requests
import pandas as pd
import streamlit as st
from tqdm import tqdm
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

# Model details 
model_file = "phi-3-sql.Q4_K_M.gguf"
model_url = f"https://huggingface.co/omeryentur/phi-3-sql/resolve/main/{model_file}"
expected_hash = "3801a2a849460675053220c9f404d714"  # MD5 hash 

# Download function with progress bar and hash check
def download_file(url, filename, expected_hash):
    cache_dir = "model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash == expected_hash:
            st.info(f"Using cached model: {filename}")
            return cache_path

    st.write(f"Downloading model: {filename}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(cache_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Re-check the hash after download
    with open(cache_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash != expected_hash:
        st.warning("Downloaded model hash does not match expected. Please re-download.")
        os.remove(cache_path)
        return None

    st.success("Download complete!")
    return cache_path

# Download and initialize LLM 
model_path = download_file(model_url, model_file, expected_hash)
if model_path:
    client = LlamaCpp(model_path=model_path, temperature=0)
else:
    st.error("Model download failed. Please try again later.")
    st.stop()

# Database Setup
db_path = "sqlite:///example.db"  
db = SQLDatabase.from_uri(database_uri=db_path)
db._sample_rows_in_table_info = 0
engine = create_engine(db_path)


# Streamlit app interface
def main():
    st.title("SQL Query Interface")

    # Display tables and contents upon page load
    display_tables_and_contents()  # Call the function to display tables

    question = st.text_input("Enter your query:", value="Courses containing Introduction")
    if st.button("Query"):
        table_info = db.get_table_info()

        template = """
            <|system|>
            {table_info}

            <|user|>
            {question}
            <|sql|>
        """

        prompt = PromptTemplate.from_template(template)
        prompt_text = prompt.format(table_info=table_info, question=question)

        res = client(prompt_text)
        sql_query = res.strip()
        print(prompt_text)

        with engine.connect() as connection:
            result = pd.read_sql_query(sql_query, connection)

        st.write(f"SQL Query: {sql_query}")
        st.write("Result:")
        st.write(result)

#Function to display tables 
def display_tables_and_contents():
    table_names = db.get_table_names()
    if table_names:
        st.write("Tables:")
        tabs = st.tabs(table_names)
        for tab, table_name in zip(tabs, table_names):
            with tab:
                st.write(f"Table: {table_name}")
                query = f"SELECT * FROM {table_name} LIMIT 5"
                with engine.connect() as connection:
                    df = pd.read_sql_query(query, connection)
                st.write(df)
    else:
        st.write("No tables found in the database.")

# Run the app
if __name__ == "__main__":
    main()
