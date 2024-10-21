from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from requests.auth import HTTPBasicAuth
import urllib3
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
import threading  # Add this line
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import time
import json
import shutil
import uuid
from chromadb import Client
from chromadb.config import Settings  # New way to configure Chroma


app = FastAPI()

# CouchDB connection parameters
COUCHDB_URL = 'https://192.168.57.185:5984'
COUCHDB_USERNAME = 'd_couchdb'
COUCHDB_PASSWORD = 'Welcome#2'
DATABASE_NAME = 'gowtham2'
GOOGLE_API_KEY = "AIzaSyAvgwBW-yBqVq3a1MjwaTDELT1inUyXSYc"

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

# Disable SSL warnings (not recommended for production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chroma database path
CHROMA_DB_PATH = "./new_chroma_db"  # Path where Chroma stores the vector data

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print(f"Old Chroma directory './chroma_db' deleted.")

# Initialize the Settings with the new path
settings = Settings(persist_directory=CHROMA_DB_PATH)

# Now create a new Chroma instance
chroma_db = Client(settings=settings)

# Ensure the Chroma directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Pydantic models for requests
class QueryRequest(BaseModel):
    query: str

class AddEmployeeRequest(BaseModel):
    doc_id: str

# Fetch all document IDs from CouchDB
def fetch_all_document_ids():
    try:
        response = requests.get(f"{COUCHDB_URL}/{DATABASE_NAME}/_all_docs?include_docs=false",
                                auth=HTTPBasicAuth(COUCHDB_USERNAME, COUCHDB_PASSWORD),
                                verify=False)
        response.raise_for_status()
        data = response.json()
        document_ids = [row['id'] for row in data.get('rows', [])]
        return document_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching document IDs: {e}")

def fetch_document(doc_id):
    try:
        response = requests.get(f"{COUCHDB_URL}/{DATABASE_NAME}/{doc_id}",
                                auth=HTTPBasicAuth(COUCHDB_USERNAME, COUCHDB_PASSWORD),
                                verify=False)
        response.raise_for_status()
        document = response.json()
        if document:
            return document
        else:
            raise ValueError(f"Document {doc_id} is empty or does not exist.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching document {doc_id}: {e}")


        

# Function to add embeddings for each document separately
def add_individual_document_embeddings(doc_id, update_type='employee'):
    try:
        # Initialize variables for additional info and leave document IDs
        additional_info_doc_id = None
        leave_doc_id = None

        # Determine which document to process based on the update_type
        if update_type == 'employee':
            # Fetch the employee document
            employee_doc = fetch_document(doc_id)
            print(f"Fetched employee document: {employee_doc}")
            additional_info_id = employee_doc.get("data", {}).get("additionalinfo_id", "")
            additional_info_doc_id = f"additionalinfo_{additional_info_id}"
            leave_doc_id = f"leave_{additional_info_id}"

            # Delete old employee embeddings
            chroma_db.delete(ids=[doc_id])
            print(f"Old embedding for {doc_id} deleted from Chroma")

            # Create new embedding for employee document
            employee_text = retrieve_employee_data(employee_doc)
            create_and_store_embedding(employee_text, doc_id)

        elif update_type == 'additional_info':

            # Fetch the additional info document
            additional_info_doc = fetch_document(doc_id)
            if not additional_info_doc:

                 print(f"Error: Document {doc_id} could not be fetched.")
                 return
            print(f"Fetched additional info document: {additional_info_doc}")

            # Delete old additional_info embeddings
            chroma_db.delete(ids=[doc_id])
            print(f"Old embedding for {doc_id} deleted from Chroma")

            # Create new embedding for additional info document
            additional_info_text = retrieve_additional_info_data(additional_info_doc)
            create_and_store_embedding(additional_info_text, doc_id)

        elif update_type == 'leave':

            # Fetch the leave document
            leave_doc = fetch_document(doc_id)
            if not leave_doc:
              print(f"Error: Document {doc_id} could not be fetched.")
              return
            print(f"Fetched leave document: {leave_doc}")

            # Delete old leave embeddings
            chroma_db.delete(ids=[doc_id])
            print(f"Old embedding for {doc_id} deleted from Chroma")

            # Create new embedding for leave document
            leave_text = retrieve_leave_data(leave_doc)
            create_and_store_embedding(leave_text, doc_id)

        else:
            raise ValueError("Invalid update_type. It should be 'employee', 'additional_info', or 'leave'.")

        # ChromaDB automatically persists changes
        print("Chroma database changes applied successfully.")

    except Exception as e:
        print(f"Error updating Chroma for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating Chroma for document {doc_id}: {e}")


# Functions to retrieve individual document data for embedding
# Fetch employee data for embedding
def retrieve_employee_data(employee_doc):
    employee_id = employee_doc.get('data', {}).get('employee_id', 'N/A')
    name = employee_doc.get('data', {}).get('name', 'N/A')
    salary = employee_doc.get('data', {}).get('salary', 'N/A')
    department = employee_doc.get('data', {}).get('department', 'N/A')
    return f"Employee ID: {employee_id}\nName: {name}\nSalary: {salary}\nDepartment: {department}"

# Fetch additional info data for embedding
def retrieve_additional_info_data(additional_info_doc):
    address = additional_info_doc.get('address', 'N/A')
    marital_status = additional_info_doc.get('martial_status', 'N/A')
    gender = additional_info_doc.get('gender', 'N/A')
    return f"Address: {address}\nMarital Status: {marital_status}\nGender: {gender}"

# Fetch leave data for embedding
def retrieve_leave_data(leave_doc):
    leave_entries = leave_doc.get('leaves', [])
    leave_dates = [leave['date'] for leave in leave_entries]
    return f"Leave Dates: {', '.join(leave_dates)}"



# Function to create and store embeddings
def create_and_store_embedding(text, doc_id):
    try:
        # Generate the new embedding using Google API
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        # Create a unique ID for the new embedding
        new_embedding_id = f"{doc_id}_{uuid.uuid4()}"

        # Add the new embedding to Chroma with a unique ID
        chroma_db.add_texts([text], metadatas=[{"doc_id": new_embedding_id}], ids=[new_embedding_id])
        print(f"New embedding for {new_embedding_id} added to Chroma")

    except Exception as e:
        print(f"Error creating or storing embedding for {doc_id}: {e}")


# Store CouchDB data in Chroma permanently
# Function to store CouchDB data in Chroma, embedding each document separately
# Function to store CouchDB data in Chroma, embedding each document separately
def store_data_in_chroma():
    try:
        # Fetch all document IDs (employee, additional_info, leave)
        document_ids = fetch_all_document_ids()

        combined_texts = []  # List of texts for embedding
        metadata = []  # Metadata for document identification

        # Loop through all document IDs and create embeddings for each type
        for doc_id in document_ids:
            if "employee_" in doc_id:
                # Fetch and embed employee document
                employee_doc = fetch_document(doc_id)
                employee_text = retrieve_employee_data(employee_doc)
                combined_texts.append(employee_text)
                metadata.append({"doc_id": doc_id, "type": "employee"})

                # Get the additional_info and leave IDs from employee data
                additional_info_id = employee_doc.get("data", {}).get("additionalinfo_id", "")
                if additional_info_id:
                    additional_info_doc_id = f"additionalinfo_{additional_info_id}"
                    leave_doc_id = f"leave_{additional_info_id}"

                    # Fetch and embed additional_info document
                    additional_info_doc = fetch_document(additional_info_doc_id)
                    additional_info_text = retrieve_additional_info_data(additional_info_doc)
                    combined_texts.append(additional_info_text)
                    metadata.append({"doc_id": additional_info_doc_id, "type": "additional_info"})

                    # Fetch and embed leave document
                    leave_doc = fetch_document(leave_doc_id)
                    leave_text = retrieve_leave_data(leave_doc)
                    combined_texts.append(leave_text)
                    metadata.append({"doc_id": leave_doc_id, "type": "leave"})

        # Store the texts and metadata in Chroma, embedding each document separately
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        chroma_db = Chroma.from_texts(
            combined_texts, 
            embeddings, 
            collection_name="couchdb_documents", 
            metadatas=metadata, 
            persist_directory=CHROMA_DB_PATH
        )
        
        # Save the vector store to disk
        chroma_db.persist()
        print("Data stored permanently in Chroma")
    
    except Exception as e:
        print(f"Error storing data in Chroma: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing data in Chroma: {e}")



from fastapi import HTTPException

from fastapi import HTTPException

def add_employee_data_to_chroma(doc_id, update_type='employee'):
    try:
        # Initialize variables for additional info and leave document IDs
        additional_info_doc_id = None
        leave_doc_id = None
        document_text = None  # Holds the document text to be embedded
        embedding_id = None    # Holds the document's embedding ID

        # Determine which document to fetch and update based on the update_type
        if update_type == 'employee':
            # Fetch the employee document
            employee_doc = fetch_document(doc_id)
            print(f"Fetched employee document: {employee_doc}")
            
            # Generate text representation of the employee document for embedding
            document_text = retrieve_employee_data(employee_doc)
            embedding_id = doc_id  # Use the employee document ID as the embedding ID

        elif update_type == 'additional_info':
            # Fetch the additional info document
            additional_info_doc_id = f"additionalinfo_{doc_id}"
            additional_info_doc = fetch_document(additional_info_doc_id)
            print(f"Fetched additional info document: {additional_info_doc}")
            
            # Generate text representation of the additional info document for embedding
            document_text = retrieve_additional_info_data(additional_info_doc)
            embedding_id = additional_info_doc_id  # Use the additional info document ID as the embedding ID

        elif update_type == 'leave':
            # Fetch the leave document
            leave_doc_id = f"leave_{doc_id}"
            leave_doc = fetch_document(leave_doc_id)
            print(f"Fetched leave document: {leave_doc}")
            
            # Generate text representation of the leave document for embedding
            document_text = retrieve_leave_data(leave_doc)
            embedding_id = leave_doc_id  # Use the leave document ID as the embedding ID

        else:
            raise ValueError("Invalid update_type. It should be 'employee', 'additional_info', or 'leave'.")

        # Generate the embedding using Google API
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        # Delete old embedding related to this document
        try:
            chroma_db.delete(ids=[embedding_id])
            print(f"Old embedding for {embedding_id} deleted from Chroma")
        except Exception as delete_err:
            print(f"Error deleting old embedding: {delete_err}")

        # Add the new embedding to Chroma with a unique ID
        chroma_db.add_texts([document_text], metadatas=[{"doc_id": embedding_id}], ids=[embedding_id])
        print(f"New embedding for {embedding_id} added to Chroma")

        # ChromaDB doesn't have a persist method, automatic persistence assumed.
        print(f"Chroma database updated successfully for {embedding_id}.")

    except Exception as e:
        print(f"Error updating Chroma for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating Chroma for document {doc_id}: {e}")












# Fetch the latest changes from CouchDB using the _changes feed with pagination
def fetch_changes_feed(since="now", limit=100):
    try:
        # Adjust the request to fetch up to 'limit' changes per request and include document data
        response = requests.get(
            f"{COUCHDB_URL}/{DATABASE_NAME}/_changes?since={since}&limit={limit}&include_docs=true&feed=longpoll",
            auth=HTTPBasicAuth(COUCHDB_USERNAME, COUCHDB_PASSWORD),
            verify=False
        )
        
        # Raise an error if the request was not successful
        response.raise_for_status()

        # Return the changes feed as a JSON response
        return response.json()

    except requests.exceptions.RequestException as req_err:
        print(f"Network-related error occurred: {req_err}")
        raise HTTPException(status_code=500, detail=f"Error fetching changes: {req_err}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching changes: {e}")
      

# Track changes in the database and log them
def monitor_couchdb_for_changes(interval=10):
    last_seq = "now"  # Track last sequence in _changes feed to avoid duplicate processing

    while True:
        try:
            changes = fetch_changes_feed(since=last_seq)
            last_seq = changes.get('last_seq', last_seq)  # Update last_seq for next request

            for change in changes.get('results', []):
                doc_id = change['id']
                revision = change['changes'][0]['rev']
                doc = change.get('doc', {})

                if not change.get('deleted', False):
                    print(f"Document {doc_id} was updated to revision {revision}.")

                    # Determine document type from doc_id and pass correct update_type to add_employee_data_to_chroma
                    if doc_id.startswith("employee_"):

                        add_individual_document_embeddings(doc_id, update_type="employee")
                    elif doc_id.startswith("additionalinfo_"):
                        add_individual_document_embeddings(doc_id, update_type="additional_info")
                    elif doc_id.startswith("leave_"):

                        add_individual_document_embeddings(doc_id, update_type="leave")
                else:
                    print(f"Document {doc_id} was deleted.")
                    # Handle the deletion of embeddings in Chroma
                    try:
                        # Initialize the Chroma database
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

                        # Delete the corresponding embedding
                        chroma_db.delete(ids=[doc_id])
                        print(f"Embedding for document {doc_id} deleted from Chroma.")
                    except Exception as delete_err:
                        print(f"Error deleting embedding for document {doc_id}: {delete_err}")

        except Exception as e:
            print(f"Error monitoring CouchDB: {e}")

        time.sleep(interval)



# Optional function to fetch the previous version of the document (before the update)
def fetch_previous_document(doc_id, current_rev):
    try:
        # Extract the numeric part and the revision hash from the current revision
        rev_num, rev_hash = current_rev.split('-')
        
        # Calculate the previous revision number
        prev_rev_num = int(rev_num) - 1

        if prev_rev_num <= 0:
            print(f"No previous revision available for document {doc_id}.")
            return None
        
        # Construct the previous revision ID
        prev_rev = f"{prev_rev_num}-{rev_hash}"

        # Request the previous revision of the document
        response = requests.get(
            f"{COUCHDB_URL}/{DATABASE_NAME}/{doc_id}?rev={prev_rev}",
            auth=HTTPBasicAuth(COUCHDB_USERNAME, COUCHDB_PASSWORD),
            verify=False
        )

        # Check if the response was successful
        if response.status_code == 200:
            return response.json()  # Return the previous document version
        else:
            print(f"Previous revision {prev_rev} not found for document {doc_id}. Status code: {response.status_code}")
            return None

    except ValueError as ve:
        # Handle cases where the revision format is incorrect or unexpected
        print(f"Invalid revision format for document {doc_id}: {current_rev}. Error: {ve}")
        return None
    except Exception as e:
        # General error handling for any other issues
        print(f"Error fetching previous document {doc_id}: {e}")
        return None



# Optional function to log differences between the previous and current document
def log_differences(old_doc, new_doc):
    def compare_docs(old, new, path=""):
        differences = {}

        # Get all keys from both documents to ensure no missing keys are left out
        all_keys = set(new.keys()).union(set(old.keys()))

        for key in all_keys:
            old_value = old.get(key)
            new_value = new.get(key)
            current_path = f"{path}/{key}" if path else key

            if isinstance(old_value, dict) and isinstance(new_value, dict):
                # Recursively compare nested dictionaries
                nested_diff = compare_docs(old_value, new_value, path=current_path)
                if nested_diff:
                    differences.update(nested_diff)

            elif old_value != new_value:
                # Log differences for non-dictionary fields
                differences[current_path] = {"old": old_value, "new": new_value}

        return differences

    if old_doc:
        # Compare the old and new documents
        differences = compare_docs(old_doc, new_doc)

        # Log the differences if any exist
        if differences:
            print(f"Differences found: {json.dumps(differences, indent=2)}")
        else:
            print("No differences found.")
    else:
        print("No previous document available for comparison.")


@app.on_event("startup")
def load_chroma_db():
    """ Load Chroma vector store on app startup. """
    try:
        global chroma_db
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        # Check if the database is empty by fetching documents
        # Replace the following line with a method from the Chroma library that checks for existing embeddings
        if hasattr(chroma_db, 'get_documents'):  # Hypothetical method
            documents = chroma_db.get_documents()  # Adjust to actual method name
            if not documents:
                print("Chroma vector store is empty.")
            else:
                print(f"Loaded {len(documents)} vectors in Chroma.")
        else:
            print("No method available to check the documents in Chroma.")

        print("Chroma vector store loaded successfully on startup")
    except Exception as e:
        print(f"Error loading Chroma vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading Chroma vector store: {e}")





@app.post("/query/")
def query_couchdb(request: QueryRequest):
    try:
        # Ensure that Chroma is loaded and available
        if chroma_db is None:
            raise HTTPException(status_code=500, detail="Chroma vector store is not loaded")

        # Create a vector retriever from the Chroma store with top 5 results
        vector_index = chroma_db.as_retriever(search_kwargs={"k": 5})

        # Initialize Google Generative AI (LLM) for query answering
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

        # Build the retrieval-based QA chain using the vector index from Chroma
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_index, return_source_documents=True)

        # Perform the query using the QA chain
        result = qa_chain({"query": request.query})

        # Extract the answer and source documents
        answer = result.get("result", "No answer found")
        source_documents = result.get("source_documents", [])

        # Return the query, the answer, and the source documents
        return {
            "query": request.query,
            "answer": answer,
            "source_documents": [doc.metadata for doc in source_documents]  # Send metadata of source documents
        }

    except HTTPException as http_err:
        # Handle known HTTP exceptions gracefully
        raise http_err

    except Exception as e:
        # Log the error and raise a 500 Internal Server Error
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.post("/add_employee/")
def add_employee_data(request: AddEmployeeRequest):
    try:
        # Ensure Chroma is loaded and available
        if chroma_db is None:
            raise HTTPException(status_code=500, detail="Chroma vector store is not loaded")

        # Dynamically add or modify employee data in Chroma
        add_employee_data_to_chroma(request.doc_id)

        # Return success message upon successful addition or update
        return {"message": f"Employee data with doc_id {request.doc_id} added/updated successfully in Chroma."}

    except HTTPException as http_err:
        # Handle known HTTP exceptions gracefully
        raise http_err

    except Exception as e:
        # Log the error and raise a 500 Internal Server Error
        print(f"Error adding/updating employee data for doc_id {request.doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding/updating employee data for doc_id {request.doc_id}: {e}")


# Initialize and start monitoring CouchDB for changes
@app.on_event("startup")
def start_monitoring_couchdb():
    try:
        # Check if Chroma vector store is loaded
        if chroma_db is None:
            raise HTTPException(status_code=500, detail="Chroma vector store is not loaded. Cannot start monitoring.")

        # Start a background thread to monitor CouchDB for changes
        monitor_thread = threading.Thread(target=monitor_couchdb_for_changes, args=(10,), daemon=True)
        monitor_thread.start()
        print("Started monitoring CouchDB for changes.")

    except HTTPException as http_err:
        # Handle known HTTP exceptions gracefully
        raise http_err

    except Exception as e:
        # Log the error and raise a 500 Internal Server Error
        print(f"Error starting CouchDB monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting CouchDB monitoring: {e}")


@app.on_event("startup")
def startup_event():
    try:
        load_chroma_db()  # Load Chroma vector store
        
        # Check if Chroma vector store is loaded
        if chroma_db is None:
            raise HTTPException(status_code=500, detail="Chroma vector store is not loaded. Cannot start monitoring.")

        # Start monitoring CouchDB for changes
        threading.Thread(target=monitor_couchdb_for_changes, args=(10,), daemon=True).start()
        print("Chroma vector store loaded successfully, and monitoring CouchDB for changes has started.")

    except HTTPException as http_err:
        # Handle known HTTP exceptions gracefully
        raise http_err

    except Exception as e:
        # Log the error and raise a 500 Internal Server Error
        print(f"Error during startup event: {e}")
        raise HTTPException(status_code=500, detail=f"Error during startup event: {e}")



# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)