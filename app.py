import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase

# Load your LLM and set up SQL database
client = LlamaCpp(model_path="phi-3-sql.Q4_K_M.gguf")
db_path = "sqlite:///example.db"
db = SQLDatabase.from_uri(database_uri=db_path)

# Define the SQL prompt template
template = """
{table_info}
{question}
"""

# Function to display tables and their contents
def display_tables_and_contents():
    table_names = db.get_table_names()
    if table_names:
        st.write("Tables:")
        st.write(table_names)
        
        
    else:
        st.write("No tables found in the database.")

# Streamlit app interface
def main():
    st.title("SQL Query Interface")
    
    # Display tables and contents upon page load
    display_tables_and_contents()
    
    question = st.text_input("Enter your query:", value="Courses containing Introduction")
    if st.button("Query"):
        # Retrieve table info
        table_info = db.get_table_info()
        
        # Create the prompt with the query
        prompt = PromptTemplate.from_template(template)
        prompt_text = prompt.format(table_info=table_info, question=question)
        
        # Get SQL query from LLM
        res = client(prompt_text)
        sql_query = res.strip()
        
        # Run SQL query and fetch result
        result = db.run(sql_query)
        
        # Display result in Streamlit app
        st.write(f"SQL Query: {sql_query}")
        st.write("Result:")
        st.write(result)

if __name__ == "__main__":
    main()
