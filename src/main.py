from langchain_core.prompts import ChatPromptTemplate
from utils import get_index, get_query_engine

def main():
    
    PROMPT_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved 
    context to answer the question. If you don't know the answer, just say that you don't know. 
    context = {context}
    question = {question}
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    dir_original_documents = input('Enter the directory where the original documents are: ')
    dir_indexed_documents = input('Enter the directory where the indexed documents are (if they already exist) or you would like to insert them: ')
    question = input("Enter search query: ")
    vectorstore = get_index(dir_original_documents, dir_indexed_documents)
    rag_chain = get_query_engine(vectorstore, prompt)
    response = rag_chain.invoke(question)
    print(response)
    

if __name__ == "__main__":
    main()