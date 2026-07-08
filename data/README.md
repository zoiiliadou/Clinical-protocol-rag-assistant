# Data Folder

This folder is used for local project data.

The clinical protocol PDF is not included in this repository due to file size and licensing considerations.

To run the project locally, download the public protocol PDF from:

https://www.nejm.org/doi/suppl/10.1056/NEJMoa2034577/suppl_file/nejmoa2034577_protocol.pdf

Then place it manually at:

data/raw/C4591001_protocol.pdf

After adding the PDF, run:

python src/ingest.py

This will generate the local ChromaDB vector store under:

data/vector_store/

The PDF file, vector store, processed files, and audit logs are intentionally excluded from GitHub.
