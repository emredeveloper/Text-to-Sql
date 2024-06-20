import os
import requests
import pandas as pd
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

# Streamlit app interface
def main():
    st.title("SQL Query Interface")

    # Display tables and contents upon page load
    display_tables_and_contents()

    question = st.text_input("Enter your query:", value="Courses containing Introduction")
    if st.button("Query"):
        # Retrieve table info
        table_info = get_table_info()

        # Define the SQL prompt template
        template = """
        {table_info}
        {question}
        """

        # Create the prompt with the query
        prompt = PromptTemplate.from_template(template)
        prompt_text = prompt.format(table_info=table_info, question=question)

        # Get SQL query from LLM
        sql_query = get_sql_query(prompt_text)

        # Run SQL query and fetch result
        with engine.connect() as connection:
            result = pd.read_sql_query(sql_query, connection)

        # Display result in Streamlit app
        st.write(f"SQL Query: {sql_query}")
        st.write("Result:")
        st.write(result)

def get_sql_query(prompt_text):
    model_file = "https://huggingface.co/omeryentur/phi-3-sql/resolve/main/phi-3-sql.Q4_K_M.gguf"
    client = LlamaCpp(model_path=model_file, temperature=0)

    res = client(prompt_text)
    sql_query = res.strip()
    return sql_query

def get_table_info():
    db_path = "sqlite:///example.db"
    db = SQLDatabase.from_uri(database_uri=db_path)
    db._sample_rows_in_table_info = 0

    table_info = db.get_table_info()
    return table_info

def display_tables_and_contents():
    db_path = "sqlite:///example.db"
    engine = create_engine(db_path)
    db = SQLDatabase.from_uri(database_uri=db_path)
    table_names = db.get_table_names()

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

if __name__ == "__main__":
    main()
