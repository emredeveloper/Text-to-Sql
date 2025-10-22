import os
import logging
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from file_utils import calculate_md5
from sql_utils import validate_insert_statement

logging.basicConfig(level=logging.ERROR)  # Set logging level

def download_file(url: str, filename: str, expected_md5: Optional[str] = None) -> None:
    """Download a file and optionally verify its integrity via MD5."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        if expected_md5:
            actual_md5 = calculate_md5(filename)
            if actual_md5 != expected_md5:
                st.error("Downloaded file failed the integrity check. Removing file.")
                os.remove(filename)
                return

        st.success("Download complete and verified!")
    except Exception as e:
        logging.exception("Error downloading model file")
        st.error("Error downloading file. Please verify your network connection and try again.")

@st.cache_resource(ttl=3600)  # Cache the model for an hour
def load_model(model_file):
    """Loads the LlamaCpp model, ensuring it's a valid .gguf file."""
    try:
        if not model_file.endswith(".gguf"):
            st.error("Invalid model file format. Please provide a .gguf file.")
            return None

        client = LlamaCpp(model_path=model_file, temperature=0)
        return client
    except Exception as e:
        logging.exception("Failed to load LlamaCpp model")
        st.error("Error loading model. Please check the model file and try again.")
        return None

def get_database():
    try:
        db_path = "sqlite:///example.db"
        db = SQLDatabase.from_uri(database_uri=db_path)
        db._sample_rows_in_table_info = 0
        engine = create_engine(db_path)
        return db, engine
    except Exception as e:
        logging.exception("Database connection failed")
        st.error("Error connecting to database. Please check the database file.")
        return None, None


def get_table_info_cached(db):
    if "table_info" not in st.session_state:
        try:
            st.session_state.table_info = db.get_table_info()
        except Exception:
            logging.exception("Failed to retrieve table information")
            st.session_state.table_info = ""
    return st.session_state.table_info

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
            selected_table = st.selectbox("Select a table to preview", table_names)
            if selected_table:
                query = f"SELECT * FROM {selected_table} LIMIT 5"
                try:
                    with engine.connect() as connection:
                        df = pd.read_sql_query(query, connection)
                    st.write(df)
                except Exception:
                    logging.exception("Failed to retrieve preview for table %s", selected_table)
                    st.error("Error retrieving table preview.")
        else:
            st.write("No tables found in the database.")

        question = st.text_area("Enter your query:", value="Courses containing Introduction")
        if st.button("Query"):
            model_file = "phi-3-sql.Q4_K_M.gguf"
            model_url = "https://huggingface.co/omeryentur/phi-3-sql/resolve/main/phi-3-sql.Q4_K_M.gguf"
            expected_md5 = None  # Provide a hash here to enable integrity verification

            # Download the model file if it doesn't exist
            if not os.path.exists(model_file):
                st.write(f"Downloading {model_file}...")
                download_file(model_url, model_file, expected_md5)

            # Load the model
            client = load_model(model_file)
            if client:
                # Retrieve table info
                table_info = get_table_info_cached(db)

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
                    with engine.connect() as connection:
                        df = pd.read_sql_query(sql_query, connection)

                    st.write(f"SQL Query: {sql_query}")
                    st.write("Result:")
                    st.write(df)
                except Exception:
                    logging.exception("Error executing generated SQL")
                    st.error("Error executing query. Please verify the generated SQL statement.")
        else:
            st.write("Please enter your query and press 'Query' to get results.")

        # Add New Data to Database section
        st.subheader("Add New Data to Database")
        new_data = st.text_area("Enter new data (SQL INSERT statement):", "")
        if st.button("Add Data"):
            if new_data.strip():
                if not validate_insert_statement(new_data):
                    st.error("Only single INSERT statements are allowed in this form.")
                else:
                    try:
                        with engine.begin() as connection:
                            connection.exec_driver_sql(new_data)
                        st.success("Data added successfully!")
                    except Exception:
                        logging.exception("Failed to execute INSERT statement")
                        st.error("Error adding data. Please verify your INSERT statement.")
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
