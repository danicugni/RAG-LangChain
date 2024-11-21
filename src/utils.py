import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document
from dotenv import load_dotenv


def set_llm():

    load_dotenv()

    llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )   
    return llm


def load_documents(dir_original_documents):
    loader = DirectoryLoader(dir_original_documents, show_progress=True)
    documents = loader.load()
    return documents


def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20, add_start_index = True)
    splits = text_splitter.split_documents(documents)
    return splits


def create_new_index(splits, dir_indexed_documents:str, embed_model:str = "sentence-transformers/all-mpnet-base-v2"):
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name=embed_model), persist_directory=dir_indexed_documents)
    return vectorstore


def load_existing_index(dir_indexed_documents:str, embed_model:str = "sentence-transformers/all-mpnet-base-v2"):
    vectorstore = Chroma(persist_directory=dir_indexed_documents, embedding_function=HuggingFaceEmbeddings(model_name = embed_model))
    return vectorstore


def get_index(dir_original_documents:str, dir_indexed_documents:str, embed_model:str = "sentence-transformers/all-mpnet-base-v2"):
    if os.path.exists(dir_indexed_documents):
        vectorstore = load_existing_index(dir_indexed_documents, embed_model)
    else:
        documents = load_documents(dir_original_documents)
        splits = split_documents(documents)
        vectorstore = create_new_index(splits, dir_indexed_documents, embed_model)
    return vectorstore


def format_docs(documents):
    return "\n\n - -\n\n".join(document.page_content for document in documents)


def get_query_engine(vectorstore, prompt, search_type:str = "similarity", search_kwargs={'k': 2}):
    llm = set_llm()
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    rag_chain = (
    {"context": retriever | format_docs, 
     "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser()
    )
    return rag_chain