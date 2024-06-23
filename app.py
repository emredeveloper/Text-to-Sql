import os
import requests
import pandas as pd
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

def download_file(url, filename):
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

# Download the model file if it doesn't exist
model_file = "phi-3-sql.Q4_K_M.gguf"
model_url = f"https://huggingface.co/omeryentur/phi-3-sql/resolve/main/{model_file}"
if not os.path.exists(model_file):
    st.write(f"Downloading {model_file}...")
    download_file(model_url, model_file)

# Cache the model loading
@st.cache_resource
def load_model():
    client = LlamaCpp(model_path=model_file, temperature=0)
    return client

# Cache the database connection
@st.cache_resource
def get_database():
    db_path = "sqlite:///example.db"
    db = SQLDatabase.from_uri(database_uri=db_path)
    db._sample_rows_in_table_info = 0
    engine = create_engine(db_path)
    return db, engine

# Cache the table names retrieval
@st.cache
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
                with engine.connect() as connection:
                    df = pd.read_sql_query(query, connection)
                st.write(df)
    else:
        st.write("No tables found in the database.")

def main():
    st.title("SQL Query Interface")

    # Retrieve database and engine
    db, engine = get_database()

    # Display tables and contents upon page load
    display_tables_and_contents(db, engine)

    question = st.text_input("Enter your query:", value="Courses containing Introduction")
    if st.button("Query"):
        # Load the model
        client = load_model()

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

        # Get SQL query from LLM
        res = client(prompt_text)
        sql_query = res.strip()
        print(prompt_text)
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, connection)

        st.write(f"SQL Query: {sql_query}")
        st.write("Result:")
        st.write(df)
    else:
        st.write("Please enter your query and press 'Query' to get results.")

if __name__ == "__main__":
    main()
