from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

os.environ["USER_AGENT"] = "Mozilla/5.0"

target_url = "https://docs.langchain.com/oss/python/langchain/rag"
jina_url = f"https://r.jina.ai/{target_url}"

loader = WebBaseLoader(
    web_paths=(jina_url,)
)

docs = loader.load()

text_xplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_xplitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))

retriever = vectorstore.as_retriever()
#rag prompt
#prompt = hub.pull("rlm/rag-prompt") HATAYI DÜZELTİNCE BU SATIRI KULLAN
prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(prompt)

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


if __name__ == '__main__':
    for chunk in rag_chain.stream("According to LangChain documentation, which of the following is the first step in the 'Indexing' phase of a RAG (Return Agility Generation) implementation?"):
        print(chunk, end="", flush=True)