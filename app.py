import os
import requests
import hashlib
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.ERROR)  # Set logging level

def calculate_md5(file_path):
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file_in_chunks(url, filename, expected_md5, num_chunks=8):
    """Downloads a file in chunks and verifies its integrity."""
    try:
        response = requests.head(url)
        file_size = int(response.headers['content-length'])
        chunk_size = file_size // num_chunks

        def download_chunk(chunk_index):
            start = chunk_index * chunk_size
            end = start + chunk_size - 1 if chunk_index < num_chunks - 1 else file_size - 1
            headers = {'Range': f'bytes={start}-{end}'}
            chunk_response = requests.get(url, headers=headers, stream=True)
            chunk_filename = f"{filename}.part{chunk_index}"
            with open(chunk_filename, 'wb') as f:
                for chunk in chunk_response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return chunk_filename

        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            chunk_files = list(executor.map(download_chunk, range(num_chunks)))

        with open(filename, 'wb') as output_file:
            for chunk_filename in chunk_files:
                with open(chunk_filename, 'rb') as chunk_file:
                    output_file.write(chunk_file.read())
                os.remove(chunk_filename)

        if calculate_md5(filename) != expected_md5:
            st.error("Downloaded file is corrupted. Please try again.")
            os.remove(filename)
        else:
            st.success("Download complete and verified!")
    except Exception as e:
        st.error(f"Error downloading file: {e}")

@st.cache_resource(ttl=3600)  # Cache the model for an hour
def load_model(model_file):
    """Loads the LlamaCpp model, ensuring it's a valid .gguf file."""
    try:
        if not model_file.endswith(".gguf"):
            st.error("Invalid model file format. Please provide a .gguf file.")
            return None

        with open(model_file, "rb") as f:
            content = f.read()
            if not content:
                st.error("Model file is empty.")
                return None

        client = LlamaCpp(model_path=model_file, temperature=0)
        return client
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_database():
    try:
        db_path = "sqlite:///example.db"
        db = SQLDatabase.from_uri(database_uri=db_path)
        db._sample_rows_in_table_info = 0
        engine = create_engine(db_path)
        return db, engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None, None

def main():
    st.title("SQL Query Interface")

    # User guide
    with st.expander("User Guide"):
        st.write("""
        This interface allows you to query an SQL database using natural language.
        - Enter your query in the input box and press 'Query' to get the results.
        - The tables and their first 5 rows are displayed upon loading the page.
        """)

    # Retrieve database and engine
    db, engine = get_database()

    if db and engine:
        # Display tables and contents upon page load
        table_names = db.get_table_names()
        if table_names:
            st.write("Tables:")
            tabs = st.tabs(table_names)
            for tab, table_name in zip(tabs, table_names):
                with tab:
                    st.write(f"Table: {table_name}")
                    query = f"SELECT * FROM {table_name} LIMIT 5"  # Limit to 5 rows for display
                    try:
                        with engine.connect() as connection:
                            df = pd.read_sql_query(query, connection)
                        st.write(df)
                    except Exception as e:
                        st.error(f"Error retrieving data from {table_name}: {e}")
        else:
            st.write("No tables found in the database.")

        question = st.text_area("Enter your query:", value="Courses containing Introduction")
        if st.button("Query"):
            model_file = "phi-3-sql.Q4_K_M.gguf"
            model_url = f"https://huggingface.co/omeryentur/phi-3-sql/resolve/main/{model_file}"
            expected_md5 = "d41d8cd98f00b204e9800998ecf8427e"  # Replace with the actual MD5 hash of the model file

            # Download the model file if it doesn't exist
            if not os.path.exists(model_file):
                st.write(f"Downloading {model_file}...")
                download_file_in_chunks(model_url, model_file, expected_md5)

            # Load the model
            client = load_model(model_file)
            if client:
                # Retrieve table info
                table_info = db.get_table_info()

                # Define the SQL prompt template
                template = """
                {table_info}
                {question}
                """

                # Create the prompt with the query
                prompt = PromptTemplate.from_template(template)
                prompt_text = prompt.format(table_info=table_info, question=question)

                try:
                    # Get SQL query from LLM
                    res = client(prompt_text)
                    sql_query = res.strip()
                    print(prompt_text)
                    with engine.connect() as connection:
                        df = pd.read_sql_query(sql_query, connection)

                    st.write(f"SQL Query: {sql_query}")
                    st.write("Result:")
                    st.write(df)
                except Exception as e:
                    st.error(f"Error executing query: {e}")
        else:
            st.write("Please enter your query and press 'Query' to get results.")

        # Option to add new data to the database
        st.subheader("Add New Data to Database")
        new_data = st.text_area("Enter new data (SQL INSERT statement):", "")
        if st.button("Add Data"):
            if new_data.strip():
                try:
                    with engine.connect() as connection:
                        connection.execute(new_data)
                    st.success("Data added successfully!")
                except Exception as e:
                    st.error(f"Error adding data: {e}")
            else:
                st.warning("Please enter a valid SQL INSERT statement.")

    else:
        st.error("Database connection not established.")

    # Button to clear cache
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")

if __name__ == "__main__":
    main()
