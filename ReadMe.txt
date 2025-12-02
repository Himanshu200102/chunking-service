Delivarables for Tuesday

Api End Points:

POST
/api/upload_folder - upload multiple documents - (eg. Apple Tax Documents - DataRoom)

GET
/api/list_all_documents - List all the Document in the folder (eg. Apple-Tax-1.pdf, Apple-Tax-2.pdf, Apple-Tax-3.pdf)

POST - 
(eg. Apple Tax Documents )
-/api/add_to_my_collection - Upload single or multiple documents to excisting DataRoom (eg. Apple Tax Documents - add more files )


GET
/api/list_all_collections - List all the DataRoom present for the organisation (eg. Apple Tax Document, Google HR Policies, etc..)


Features -

Company employees can create multiple DataRoom 
2 - Upload multiple files 
3 - List all files 
4 - update individual files 
5 - Delete files    


Usecase:

1. Seach from all the documents present in DataRoom
2. Summarize or Q&A with Individual files in the DataRoom



# Basic Api-end points created - should be replaced with (RAG implimentation)

Create a new DataRoom and upload multiple files
List all existing DataRooms
List all files inside a specific DataRoom
Add new files to an existing DataRoom
Update (replace) individual files
Delete individual files
Delete an entire DataRoom


## To Run Server
source dataroom/bin/activate       
dataroom\Scripts\activate  
uvicorn app.main:app --reload


## To-DO-List - Make sure moving on the different features each Module should be tested to make sure its giving the exact output
Friday (Today)

1. Ingestion – Set up and test document ingestion pipeline for all supported file types (.pdf, .docx, .xlsx, images).
2. Chunking – Implement structure-aware and token-based chunking logic (maintain headers, tables, lists).
Deadline: Friday EOD

Saturday
3. VectorDB Setup – Integrate LanceDB / OpenSearch hybrid for vector and sparse indexing.
4. Retrieval – Implement hybrid retrieval logic (dense + sparse search) and reranking with metadata context.


Sunday
5. LLM Integration – Connect retrieval output to local/offline LLM for context-aware responses.
6. Response Generation – Test and refine answer formatting, citations, and accuracy metrics.

## batch processing
## max 2 projects for one user - 20 each max --> if condition is not matched throw error
## stream responses