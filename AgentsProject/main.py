from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

memory = MemorySaver()

search = TavilySearchResults(max_results=2)

tools = [search]

agent_executor = create_react_agent(model, tools,checkpointer=memory)

config = {"configurable": {"thread_id":"abc123"}}

if __name__ == "__main__":
    while True:
        user_input = input("> ")
        for chunk in agent_executor.stream(
                {"messages":[HumanMessage(content=user_input)]},
            config=config
        ):
            print(chunk)
            print("---")