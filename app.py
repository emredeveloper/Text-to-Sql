import pandas as pd
import streamlit as st
from transformers import AutoModelForCausalLM, pipeline
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
        template = f"""
        {table_info}
        {question}
        """

        # Get SQL query from LLM using Hugging Face transformer
        sql_query = get_sql_query(template)

        # Run SQL query and fetch result
        with engine.connect() as connection:
            result = pd.read_sql_query(sql_query, connection)

        # Display result in Streamlit app
        st.write(f"SQL Query: {sql_query}")
        st.write("Result:")
        st.write(result)

def get_sql_query(prompt_text):
    # Load Hugging Face transformer model
    model = AutoModelForCausalLM.from_pretrained("omeryentur/phi-3-sql")
    generator = pipeline(task="text-generation", model=model, device=0)

    # Generate SQL query using Hugging Face model
    response = generator(prompt_text, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=3)
    sql_query = response[0]["generated_text"].strip()
    return sql_query

def get_table_info():
    db_path = "sqlite:///example.db"
    db = SQLDatabase.from_uri(database_uri=db_path)
    db._sample_rows_in_table_info = 0

    table_info = db.get_table_info()
    return table_info

def display_tables_and_contents():
    db_path = "sqlite:///example.db"
    global engine 
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
