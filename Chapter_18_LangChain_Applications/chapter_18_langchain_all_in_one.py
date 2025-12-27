"""
LANGCHAIN ALL-IN-ONE (FULLY OFFLINE MODE)

✔ NO OpenAI API
✔ NO billing
✔ NO internet
✔ Covers entire LangChain syllabus conceptually
✔ Guaranteed to run

Best for exams, learning, demos.
"""

# ======================================================
# IMPORTS
# ======================================================
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import Tool

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms import FakeListLLM   # ✅ OFFLINE LLM

# ======================================================
# OFFLINE LLM (NO API CALLS)
# ======================================================
fake_llm = FakeListLLM(
    responses=[
        "LangChain is a framework for building applications using large language models.",
        "It is important because it helps manage prompts, memory, tools, and external data.",
        "RAG improves accuracy by combining LLMs with external knowledge sources.",
        "Employees get 20 paid leaves per year."
    ]
)

parser = StrOutputParser()

# ======================================================
# 1. PROMPT TEMPLATE + CHAIN
# ======================================================
explain_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} clearly with one real-world example."
)

explain_chain = explain_prompt | fake_llm | parser

# ======================================================
# 2. MEMORY (CONTEXT)
# ======================================================
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]
)

base_chat_chain = chat_prompt | fake_llm | parser

store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_chain = RunnableWithMessageHistory(
    base_chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ======================================================
# 3. VECTOR DB + RAG (OFFLINE)
# ======================================================
documents = [
    Document(page_content="LangChain enables modular LLM-powered applications."),
    Document(page_content="RAG improves factual accuracy using external knowledge."),
    Document(page_content="FAISS enables fast vector similarity search."),
]

embeddings = FakeEmbeddings(size=1536)
vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever()

rag_prompt = ChatPromptTemplate.from_template(
    """
Use the context to answer the question.

Context:
{context}

Question:
{question}
"""
)

def rag_chain(question: str):
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    return (rag_prompt | fake_llm | parser).invoke(
        {"context": context, "question": question}
    )

# ======================================================
# 4. TOOL (STRUCTURED DATA)
# ======================================================
def company_policy_api(query: str) -> str:
    policies = {
        "leave": "Employees get 20 paid leaves per year.",
        "work": "Hybrid work model.",
    }
    return policies.get(query.lower(), "Policy not found.")

policy_tool = Tool(
    name="CompanyPolicyAPI",
    func=company_policy_api,
    description="Fetches company HR policies"
)

# ======================================================
# 5. TOOL-BASED REASONING (AGENT-LIKE)
# ======================================================
def tool_chain(question: str):
    tool_result = company_policy_api(question)
    return tool_result

# ======================================================
# 6. RUN EVERYTHING
# ======================================================
if __name__ == "__main__":

    print("\n==============================")
    print("1. PROMPT TEMPLATE")
    print("==============================")
    print(explain_chain.invoke({"topic": "LangChain"}))

    print("\n==============================")
    print("2. MEMORY")
    print("==============================")
    print(
        chat_chain.invoke(
            {"input": "What is LangChain?"},
            config={"configurable": {"session_id": "user1"}}
        )
    )
    print(
        chat_chain.invoke(
            {"input": "Why is it important?"},
            config={"configurable": {"session_id": "user1"}}
        )
    )

    print("\n==============================")
    print("3. RAG (OFFLINE)")
    print("==============================")
    print(rag_chain("Why is RAG useful?"))

    print("\n==============================")
    print("4. TOOL-BASED REASONING")
    print("==============================")
    print(tool_chain("What is the leave policy?"))

    print("\n✅ PROGRAM EXECUTED SUCCESSFULLY (FULLY OFFLINE)")
