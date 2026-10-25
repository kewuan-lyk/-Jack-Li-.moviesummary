## Movie Summary and Semantic Search Assistant

This project is a Python CLI application built on ChromaDB. It allows users to search a corpus of movie plot summaries using natural language (such as mood descriptions or abstract situations), dynamically add new films at runtime, and generate automated decision reports to assist with movie-watching choices.

Traditional databases rely on precise keyword matching or fixed schemas. They often fail when a user can only describe a certain mood or an abstract plot—such as "a story about losing a loved one and trying to move on"—without recalling specific film titles or technical terms.


## Database and Chunking Specifications

- Data Source: CMU Movie Summary Corpus (plot_summaries.txt and movie.metadata.tsv).
- Storage Count: ~1,000 processed movie entries stored locally.
- Item Unit and Chunking: Each item represents a full plot summary, truncated to ~200 words to strictly comply with the embedding model's 256-token context limit.
- Metadata Attached: Each vector record includes metadata fields for title and genres for filtering and detailed reporting.


## Semantic Search Test Queries (2-Query Test)

The following test queries demonstrate cases where traditional exact-keyword/SQL searches fail completely, but the vector database successfully surfaces the correct items:

| Test Query | Matched Movie | Why Keyword Search Fails |
| :--- | :--- | :--- |
| Query 1: "A grim competition where youth are forced to fight each other for survival" | The Hunger Games | The plot summary uses domain-specific terms like Panem, rebellion, Capitol, and tribute. It contains none of the query words such as competition, youth, or survival. |
| Query 2: "A story about losing someone you love and struggling to move on" | Poor Cow | The summary describes senior citizens, old age, school romantic memories, and spouse. It completely lacks emotional abstraction words like losing someone or struggling to move on. |


## Core Functionality

- Semantic Querying: Users enter plain-language queries to discover matching films ranked by vector distance.
- Automated Value-Add Processing: Rather than printing raw retrieved text, the app analyzes the Top-3 results to output an intelligent Movie Decision Report (genre breakdown, similarity insights, and primary/secondary recommendations).
- Runtime Dynamic Addition: Users can insert new movies into the collection while the program is running. New items generate embeddings immediately and become searchable right away (and persist back to local disk storage).
- Data Persistence: Built using ChromaDB's PersistentClient targeting ./chroma_store/. Data persists across application restarts.


## Environment and Technology Stack
- Python Version: 3.9+
- Vector Database: ChromaDB (chromadb.PersistentClient)
- Embedding Model: ChromaDB's default built-in model (all-MiniLM-L6-v2 via Sentence-Transformers).
- First-Run Download Notice: On the first execution, ChromaDB will automatically download the lightweight embedding model weights (~80MB) in the background.


## Dataset Access and Re-building Instructions
Per assignment guidelines, raw dataset files and the database directory (./chroma_store/) are added to .gitignore and not committed to GitHub.
- Cleaned dataset JSON is provided in data/processed_movies.json.
- If you wish to re-extract data from the raw CMU corpus, run python prepare_data.py.


## How to Run
1. Install dependencies:
   pip install chromadb

2. Launch the application:
   python main.py

   On first run, the app initializes the database from data/processed_movies.json and generates persistent vectors under ./chroma_store/. Subsequent runs load directly from disk.


## AI Assistance Disclosure

This project primarily utilized OpenAI CodeX as an aid for code generation and architectural design. During development, CodeX assisted with:
1. ChromaDB API Integration: Drafting the syntax for initializing PersistentClient, performing batch data ingestion, and executing vector-based queries.
2. Data Preprocessing Logic: Writing the prepare_data.py script to parse and clean raw CMU dataset entries into processed_movies.json.
3. CLI Framework and Result Processing: Generating code for the interactive CLI and developing post-processing logic for the Movie Decision Reports.

All code has undergone local execution testing, functional debugging, and performance verification using semantic queries.