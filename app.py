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

def download_file(url, filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise error for bad status codes
            total_length = int(r.headers.get('content-length', 0))
            with open(filename, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = downloaded / total_length
                        st.progress(progress)
        st.success("Download complete!")
    except requests.exceptions.HTTPError as e:
        st.exception(f"HTTP error occurred while downloading file: {e}")
    except Exception as e:
        st.exception(f"Error downloading file: {e}")

def load_model(model_file):
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
        st.exception(f"Error loading model: {e}")
        return None

def get_database():
    try:
        db_path = "sqlite:///example.db"
        db = SQLDatabase.from_uri(database_uri=db_path)
        db._sample_rows_in_table_info = 0
        engine = create_engine(db_path)
        return db, engine
    except Exception as e:
        st.exception(f"Error connecting to database: {e}")
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
        display_tables_and_contents(db, engine)

        question = st.text_input("Enter your query:", value="Courses containing Introduction")
        if st.button("Query"):
            model_file = "phi-3-sql.Q4_K_M.gguf"
            model_url = f"https://huggingface.co/omeryentur/phi-3-sql/resolve/main/{model_file}"
            expected_md5 = "d41d8cd98f00b204e9800998ecf8427e"  # Replace with the actual MD5 hash of the model file

            # Download the model file if it doesn't exist
            if not os.path.exists(model_file):
                st.write(f"Downloading {model_file}...")
                download_file(model_url, model_file)

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
                    st.exception(f"Error executing query: {e}")
        else:
            st.write("Please enter your query and press 'Query' to get results.")
    else:
        st.error("Database connection not established.")

    # Button to clear cache
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")

if __name__ == "__main__":
    main()
