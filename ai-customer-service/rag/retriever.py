import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



VECTOR_PATH = "./chroma_db"



def search_policy(question):

    embeddings = HuggingFaceEmbeddings(
        model_name=
        "BAAI/bge-small-zh-v1.5"
    )


    db = Chroma(
        persist_directory=VECTOR_PATH,
        embedding_function=embeddings
    )


    docs = db.similarity_search(
        question,
        k=3
    )


    return "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )



if __name__ == "__main__":

    result = search_policy(
        "已经发货还能退款吗？"
    )

    print(result)