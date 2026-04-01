from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


class LocalRetriever:
    """Simple offline retriever used when external services are unavailable."""

    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents

    def invoke(self, question: str) -> List[Document]:
        q = question.lower()
        hits = [d for d in self.documents if any(token in d.page_content.lower() for token in q.split())]
        return hits if hits else self.documents[:2]


retriever = LocalRetriever(
    [
        Document(page_content="RAG combines retrieval and generation to ground LLM answers in source documents."),
        Document(page_content="Vector databases support semantic search for relevant context retrieval."),
        Document(page_content="Web search can complement vectorstore retrieval when local documents are insufficient."),
    ]
)