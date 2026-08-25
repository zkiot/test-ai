from pathlib import Path
import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOC_PATH = Path(
    "./docs/refund_policy.md"
)

VECTOR_PATH = "./chroma_db"


def create_vector_store():
    loader = TextLoader(
        str(DOC_PATH),
        encoding="utf-8"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=
        "BAAI/bge-small-zh-v1.5"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_PATH
    )

    print(
        f"向量库创建完成，共{len(chunks)}个chunk"
    )


if __name__ == "__main__":
    create_vector_store()
