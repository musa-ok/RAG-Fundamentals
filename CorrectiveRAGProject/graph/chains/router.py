import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

class RouterQuery(BaseModel):
    """
    Router a user query to the most relevant datasource
    """

    datasource : Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore",
    )

system_prompt = """
You are an expert at routing a user question to a vectorstore or web search. 
The vectorstore contains specialized documents related to:
1. The original RAG framework and architectural foundations (Lewis et al.).
2. Production-grade RAG implementation strategies, including hybrid search and semantic caching.
3. Scaling RAG systems for real-world enterprise applications in 2026.
4. Technical concepts of vector databases and semantic retrieval.
Use the vectorstore for questions on these topics. For all else, use web-search
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
     ]
)

if os.getenv("GOOGLE_API_KEY"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm_router = llm.with_structured_output(RouterQuery)
    question_router = route_prompt | structured_llm_router
else:
    question_router = RunnableLambda(lambda _: RouterQuery(datasource="vectorstore"))
