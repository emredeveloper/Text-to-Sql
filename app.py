import os
import requests
import pandas as pd
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

def download_file(url, filename):
    try:
        with requests.get(url, stream=True) as r:
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
    except Exception as e:
        st.error(f"Error downloading file: {e}")

# Cache the model loading
@st.cache_resource(ttl=3600)
def load_model(model_file):
    try:
        client = LlamaCpp(model_path=model_file, temperature=0)
        return client
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Cache the database connection
@st.cache_resource(ttl=3600)
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

# Cache the table names retrieval
@st.cache(ttl=600)
def get_table_names(db):
    return db.get_table_names()

def display_tables_and_contents(db, engine):
    table_names = get_table_names(db)
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

        question = st.text_area("Enter your query:", value="Courses containing Introduction")
        if st.button("Query"):
            model_file = "phi-3-sql.Q4_K_M.gguf"
            model_url = f"https://huggingface.co/omeryentur/phi-3-sql/resolve/main/{model_file}"

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
                    st.error(f"Error executing query: {e}")
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
