from pathlib import Path

import requests
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
current_dir = Path(__file__).resolve().parent

DB_NAME = Path(current_dir, "faiss_example_index")
TEST_FILE_URLS = [
    "https://en.wikipedia.org/api/rest_v1/page/pdf/Artificial_general_intelligence",
    "https://en.wikipedia.org/api/rest_v1/page/pdf/Socrates",
    "https://en.wikipedia.org/api/rest_v1/page/pdf/Seychelles_parakeet",
]
PDF_FOLDER = Path(current_dir, "example_documents")


def download_pdf(url: str, dst: Path) -> None:
    """Downloads a PDF file from a given URL if it doesn't exist in the destination directory"""
    dst_path = dst / f"{url.split('/')[-1].replace(' ', '_')}.pdf"

    if not dst_path.is_file():
        print("Downloading", url)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url)
        response.raise_for_status()

        with dst_path.open("wb") as f:
            f.write(response.content)


def download_and_load_documents(pdfs_path: Path, urls: list[str]):
    """Downloads PDFs from a list of URLs and loads them into a list of documents"""
    for url in urls:
        download_pdf(url, pdfs_path)

    loader = DirectoryLoader(str(pdfs_path), glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def set_up_faiss_db(db_path: Path, pdfs_path: Path, chunk_size=420, chunk_overlap=30):
    """Setups up a Faiss DB by loading documents and creating an index"""

    if db_path.exists():
        try:
            return FAISS.load_local(str(db_path), embeddings)
        except Exception as e:
            print(f"Failed to load local FAISS DB: {e}")
            raise

    documents = download_and_load_documents(pdfs_path, TEST_FILE_URLS)
    if not documents:
        raise ValueError(f"No documents loaded from {pdfs_path}")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(str(db_path))

    return db


def initiate_test_faiss():
    """Initiates a Faiss test by creating a RetrievalQA object"""
    db = set_up_faiss_db(DB_NAME, PDF_FOLDER)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3})
    )
    return qa
