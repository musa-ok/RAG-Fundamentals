import os

import langchainhub as hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

if os.getenv("GOOGLE_API_KEY"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = hub.pull("rlm/rag-prompt")
    generation_chain = prompt | llm | StrOutputParser()
else:
    generation_chain = RunnableLambda(
        lambda x: f"Mock answer: {x['question']} | Context docs: {len(x['context'])}"
    )
