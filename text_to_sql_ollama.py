import pandas as pd
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts.prompt import PromptTemplate
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
import logging
import datetime
from streamlit_ace import st_ace
import json
import sqlite3

from sql_utils import validate_insert_statement

logging.basicConfig(level=logging.ERROR)  # Set logging level

@st.cache_resource(ttl=3600)  # Cache the model for an hour
def load_ollama_model(model_name="gemma3"):
    """Loads the Ollama model"""
    try:
        client = OllamaLLM(
            model=model_name,
            temperature=0,
            base_url="http://localhost:11434",  # Default Ollama API URL
            timeout=120  # Increase timeout for longer responses
        )
        return client
    except Exception as e:
        logging.exception("Failed to connect to Ollama")
        st.error("Error connecting to Ollama. Ensure the service is running and try again.")
        st.info("Make sure Ollama is running on your system with the gemma3 model installed.")
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
    if "ollama_table_info" not in st.session_state:
        try:
            st.session_state.ollama_table_info = db.get_table_info()
        except Exception:
            logging.exception("Failed to retrieve table information")
            st.session_state.ollama_table_info = ""
    return st.session_state.ollama_table_info

def export_database(db_path, export_format):
    try:
        conn = sqlite3.connect(db_path.replace('sqlite:///', ''))
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if export_format == 'CSV':
            # Create a ZIP file with CSVs
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                for table in tables:
                    table_name = table[0]
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(f"{table_name}.csv", csv_data)
            
            return zip_buffer.getvalue(), 'application/zip', f"database_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
        elif export_format == 'JSON':
            result = {}
            for table in tables:
                table_name = table[0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                result[table_name] = json.loads(df.to_json(orient='records'))
                
            return json.dumps(result, indent=4).encode(), 'application/json', f"database_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        elif export_format == 'SQL':
            # Generate SQL dump
            sql_dump = ""
            for table in tables:
                table_name = table[0]
                # Get CREATE TABLE statement
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
                create_stmt = cursor.fetchone()[0]
                sql_dump += f"{create_stmt};\n\n"
                
                # Get INSERT statements
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                for _, row in df.iterrows():
                    columns = ', '.join(df.columns)
                    values = ', '.join([f"'{str(x).replace("'", "''")}'" if isinstance(x, str) else str(x) if x is not None else 'NULL' for x in row])
                    sql_dump += f"INSERT INTO {table_name} ({columns}) VALUES ({values});\n"
                sql_dump += "\n"
                
            return sql_dump.encode(), 'text/plain', f"database_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        return None, None, None
        
    except Exception:
        logging.exception("Failed to export database")
        st.error("Error exporting database. Please try again.")
        return None, None, None

def main():
    st.title("Text-to-SQL with Ollama")
    
    # Initialize session state for query history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Sidebar for model selection and additional features
    with st.sidebar:
        st.header("Model Configuration")
        model_name = st.selectbox(
            "Select Ollama Model",
            ["gemma3", "gemma3:7b", "gemma3:2b", "llama3", "phi3", "mistral"],
            index=0
        )
        
        st.info(
            "Make sure the selected model is installed in your Ollama.\n\n"
            "To install a model in Ollama, run:\n"
            "`ollama pull " + model_name + "`"
        )
        
        # Export database section
        st.header("Database Tools")
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "SQL"],
            index=0
        )
        
        if st.button("Export Database"):
            db_path = "sqlite:///example.db"
            export_data, mime_type, filename = export_database(db_path, export_format)
            if export_data:
                st.download_button(
                    label="Download Export",
                    data=export_data,
                    file_name=filename,
                    mime=mime_type
                )
        
        # Clear cache button
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
            
        # Clear history button
        if st.button("Clear Query History"):
            st.session_state.query_history = []
            st.success("Query history cleared!")
    
    # User guide
    with st.expander("User Guide"):
        st.write("""
        This interface allows you to query an SQL database using natural language.
        - Enter your query in the input box and press 'Query' to get the results.
        - The tables and their first 5 rows are displayed upon loading the page.
        - Uses Ollama with LLM models for text-to-SQL conversion.
        - You can also write and execute SQL queries directly using the SQL Editor tab.
        - Query history is saved during your session.
        - Export your database in various formats using the sidebar tools.
        - Make sure you have Ollama installed and running on your system.
        """)

    # Retrieve database and engine
    db, engine = get_database()

    if db and engine:
        # Create tabs for Natural Language vs Direct SQL
        tab1, tab2, tab3 = st.tabs(["Natural Language Query", "SQL Editor", "Query History"])
        
        with tab1:
            # Display tables and contents upon page load
            table_names = db.get_usable_table_names()
            if table_names:
                st.write("Tables:")
                selected_table = st.selectbox("Select a table to preview", table_names, key="ollama_table_preview")
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
            
            if st.button("Query", type="primary"):
                with st.spinner("Connecting to Ollama and generating SQL query..."):
                    # Load the Ollama model
                    client = load_ollama_model(model_name)
                    
                    if client:
                        # Retrieve table info
                        table_info = get_table_info_cached(db)
                        
                        # Define the SQL prompt template
                        template = """
                        You are an SQL expert. I'll provide you with database schema information and a question. 
                        Your task is to generate the correct SQL query to answer the question based on the schema.
                        
                        Database schema:
                        {table_info}
                        
                        Question: {question}
                        
                        Respond with the SQL query only, no explanations.
                        """
                        
                        # Create the prompt with the query
                        prompt = PromptTemplate.from_template(template)
                        prompt_text = prompt.format(table_info=table_info, question=question)
                        
                        try:
                            # Get SQL query from Ollama
                            st.info("Generating SQL query...")
                            res = client.invoke(prompt_text)
                            
                            # Extract SQL from the response
                            sql_query = res.strip()
                            
                            # Try to clean the query if it contains markdown formatting
                            if "```sql" in sql_query:
                                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
                            elif "```" in sql_query:
                                sql_query = sql_query.split("```")[1].split("```")[0].strip()
                            
                            # Display the generated SQL
                            st.code(sql_query, language="sql")
                            
                            # Execute the query and show results
                            with engine.connect() as connection:
                                try:
                                    df = pd.read_sql_query(sql_query, connection)
                                    st.success("Query executed successfully!")
                                    st.write("Result:")
                                    st.write(df)

                                    # Add to query history
                                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.query_history.append({
                                        "timestamp": timestamp,
                                        "natural_language": question,
                                        "sql_query": sql_query,
                                        "result_count": len(df)
                                    })
                                except Exception as sql_error:
                                    logging.exception("Failed to execute generated SQL")
                                    st.error("Error executing SQL. Please review the generated statement.")
                                    st.warning("The generated SQL may have syntax errors. You can try rephrasing your question.")
                        except Exception as e:
                            logging.exception("Failed to generate SQL from Ollama")
                            st.error("Error generating query. Please ensure Ollama is running and try again.")

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
        
        with tab2:
            st.subheader("SQL Editor")
            st.write("Write and execute SQL queries directly")
            
            # Use streamlit_ace for SQL editing with syntax highlighting
            sql_code = st_ace(
                placeholder="Enter your SQL query here...",
                language="sql",
                theme="github",
                keybinding="vscode",
                font_size=14,
                height=200
            )
            
            if st.button("Execute SQL"):
                if sql_code.strip():
                    try:
                        with engine.connect() as connection:
                            df = pd.read_sql_query(sql_code, connection)
                            st.success("Query executed successfully!")
                            st.write("Result:")
                            st.write(df)

                            # Add to query history
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.query_history.append({
                                "timestamp": timestamp,
                                "natural_language": "Direct SQL Input",
                                "sql_query": sql_code,
                                "result_count": len(df)
                            })
                    except Exception:
                        logging.exception("Failed to execute manual SQL")
                        st.error("Error executing SQL. Please review your statement.")
                else:
                    st.warning("Please enter a SQL query.")
        
        with tab3:
            st.subheader("Query History")
            if st.session_state.query_history:
                for i, query in enumerate(reversed(st.session_state.query_history)):
                    with st.expander(f"{query['timestamp']} - {query['natural_language'][:50]}..."):
                        st.write(f"Natural Language: {query['natural_language']}")
                        st.code(query['sql_query'], language="sql")
                        st.write(f"Result count: {query['result_count']} rows")
                        
                        if st.button(f"Re-run Query #{i}"):
                            try:
                                with engine.connect() as connection:
                                    df = pd.read_sql_query(query['sql_query'], connection)
                                    st.success("Query executed successfully!")
                                    st.write("Result:")
                                    st.write(df)
                            except Exception:
                                logging.exception("Failed to re-run query from history")
                                st.error("Error executing SQL. Please review the stored statement.")
            else:
                st.info("No queries in history yet.")

    else:
        st.error("Database connection not established.")
        
        # Create database if it doesn't exist
        if st.button("Create Example Database"):
            try:
                # Create a new SQLite database
                engine = create_engine("sqlite:///example.db")
                with engine.connect() as connection:
                    # Create courses table
                    connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS courses (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT
                    )
                    """))
                    
                    # Create students table
                    connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        age INTEGER,
                        course_id INTEGER,
                        FOREIGN KEY (course_id) REFERENCES courses(id)
                    )
                    """))
                    
                    # Insert sample data
                    connection.execute(text("""
                    INSERT INTO courses (id, name, description) VALUES
                    (1, 'Mathematics', 'Basic Mathematics Course'),
                    (2, 'Physics', 'Introduction to Physics'),
                    (3, 'Chemistry', 'Basic Chemistry Course'),
                    (4, 'Biology', 'Introduction to Biology'),
                    (5, 'History', 'World History'),
                    (6, 'Geography', 'Introduction to Geography'),
                    (7, 'Literature', 'English Literature'),
                    (8, 'Art', 'Visual Arts'),
                    (9, 'Computer Science', 'Introduction to Computer Science'),
                    (10, 'Music', 'Music Theory')
                    """))
                    
                    connection.execute(text("""
                    INSERT INTO students (id, name, age, course_id) VALUES
                    (1, 'John Smith', 20, 1),
                    (2, 'Emily Johnson', 21, 2),
                    (3, 'Michael Brown', 19, 3),
                    (4, 'Emma Davis', 22, 4),
                    (5, 'James Wilson', 20, 5),
                    (6, 'Olivia Martinez', 21, 6),
                    (7, 'William Anderson', 19, 7),
                    (8, 'Sophia Thomas', 22, 8),
                    (9, 'Benjamin Taylor', 20, 9),
                    (10, 'Isabella Hernandez', 21, 10)
                    """))
                    
                    connection.commit()
                
                st.success("Example database created successfully! Please refresh the page.")
            except Exception:
                logging.exception("Failed to create example database")
                st.error("Error creating database. Please check file permissions and try again.")

if __name__ == "__main__":
    main()