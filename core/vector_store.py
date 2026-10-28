from pathlib import Path

from core.transcriber import transcribe_source
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")

COLLECTION = "video_knowledge"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def add_transcript(transcript, source):
    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": source, "type": "transcript", "chunk": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    ids = [f"{source}-transcript-{i}" for i in range(len(docs))]

    store.add_documents(docs, ids=ids)

    return len(docs)


def get_retriever(k=4):
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def ingest_source(source, language=None, translate=False):

    transcript = transcribe_source(source, language=language, translate=translate)

    return add_transcript(transcript, source)
