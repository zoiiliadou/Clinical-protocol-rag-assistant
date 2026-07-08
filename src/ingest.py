import logging
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PDF_PATH = Path("data/raw/C4591001_protocol.pdf")
CHROMA_PATH = Path("data/vector_store")
COLLECTION_NAME = "clinical_protocol"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    logger.info("Starting data ingestion process for C4591001 clinical protocol.")

    if not PDF_PATH.exists():
        logger.error(f"PDF file not found: {PDF_PATH}")
        logger.error("Make sure the protocol PDF is inside data/raw/ and has this exact name.")
        return

    if CHROMA_PATH.exists():
        logger.info(f"Removing old vector store: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    logger.info("Loading PDF document...")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        logger.error("No pages were loaded from the PDF.")
        return

    logger.info(f"Successfully loaded {len(documents)} pages.")

    for doc in documents:
        doc.metadata["source"] = PDF_PATH.name
        doc.metadata["document_type"] = "clinical_protocol"

    logger.info("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        logger.error("No chunks were created from the PDF.")
        return

    logger.info(f"Document split into {len(chunks)} text chunks.")

    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info("Creating new ChromaDB vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME
    )

    logger.info(f"Vector store successfully created at: {CHROMA_PATH}")
    logger.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
