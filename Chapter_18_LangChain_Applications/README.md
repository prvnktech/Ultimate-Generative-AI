# Chapter 18: LangChain Applications

This chapter explores LangChain, a powerful framework for building applications with Large Language Models (LLMs), enabling complex workflows, chains, agents, and memory systems.

## Overview

LangChain simplifies the development of LLM-powered applications by providing modular components that can be combined to create sophisticated AI systems. This chapter covers:

- **LangChain Fundamentals** - Core concepts and architecture
- **Chains** - Sequential processing pipelines
- **Agents** - Autonomous decision-making systems
- **Memory** - Maintaining conversation context
- **Tools** - Extending LLM capabilities
- **Retrieval-Augmented Generation (RAG)** - Combining LLMs with knowledge bases

## Features

- **All-in-One Notebook**: Comprehensive LangChain demonstrations
- **Multiple Use Cases**: Chatbots, QA systems, agents, RAG
- **Practical Examples**: Real-world application patterns
- **Integration Guides**: Connect with APIs, databases, and tools
- **Best Practices**: Production-ready patterns

## Files

- `chapter_18_langchain_all_in_one.py` - Complete LangChain implementation
- `Notebook/chapter_18_langchain_all_in_one.ipynb` - Detailed interactive notebook

## Installation

```bash
pip install langchain langchain-community langchain-openai
```

Additional dependencies:
```bash
pip install openai chromadb faiss-cpu tiktoken  # Vector stores & embeddings
pip install google-search-results wikipedia     # External tools
pip install sentence-transformers               # Local embeddings
```

## Usage

### Python Script

```bash
python Chapter_18_LangChain_Applications/chapter_18_langchain_all_in_one.py
```

### Jupyter Notebook

```bash
jupyter notebook Chapter_18_LangChain_Applications/Notebook/chapter_18_langchain_all_in_one.ipynb
```

## Key Concepts

### LangChain Architecture

```
Application Layer
    ├── Chains (Sequential workflows)
    ├── Agents (Decision-making systems)
    ├── Memory (Context management)
    └── Tools (External capabilities)
         
Core Components
    ├── LLMs (Language models)
    ├── Prompts (Templates)
    ├── Document Loaders (Data ingestion)
    ├── Vector Stores (Semantic search)
    └── Embeddings (Text representations)
```

### 1. Models

**LLM Integration:**
```python
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

# OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Hugging Face (local)
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation"
)

# Use the model
response = llm.invoke("What is LangChain?")
```

### 2. Prompts

**Prompt Templates:**
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Simple template
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}"
)

# Chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me about {topic}")
])

# Use template
prompt = template.format(topic="artificial intelligence")
```

### 3. Chains

**Sequential Processing:**
```python
from langchain.chains import LLMChain, SimpleSequentialChain

# Single chain
chain = LLMChain(llm=llm, prompt=template)
result = chain.run(topic="machine learning")

# Sequential chains
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
sequential_chain = SimpleSequentialChain(chains=[chain1, chain2])
result = sequential_chain.run("Start with this")
```

### 4. Memory

**Conversation Context:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# Add messages
memory.save_context(
    {"input": "Hi, I'm Alice"},
    {"output": "Hello Alice! How can I help you?"}
)

# Retrieve history
history = memory.load_memory_variables({})
```

### 5. Agents

**Autonomous Decision-Making:**
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Useful for math calculations"
    )
]

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run agent
result = agent_executor.invoke({"input": "What is 25 * 4?"})
```

## What You'll Learn

1. **LangChain Basics**
   - Core components and abstractions
   - Model wrappers and providers
   - Prompt engineering with templates
   - Chain composition patterns

2. **Advanced Features**
   - Agents and tools
   - Memory systems
   - Retrieval-Augmented Generation (RAG)
   - Document processing

3. **Real-World Applications**
   - Chatbots with memory
   - Question-answering systems
   - Research assistants
   - Code generation tools

4. **Production Patterns**
   - Error handling
   - Monitoring and logging
   - Cost optimization
   - Deployment strategies

## Common Use Cases

### 1. Simple Chatbot

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat
conversation.predict(input="Hi, my name is Alice")
conversation.predict(input="What's my name?")  # Remembers!
```

### 2. Question Answering over Documents

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load and split documents
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask questions
answer = qa.run("What is the main topic?")
```

### 3. Agent with Tools

```python
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

# Load tools
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use agent
result = agent.run(
    "What is the population of Tokyo in 2023 multiplied by 2?"
)
```

### 4. Summarization Chain

```python
from langchain.chains.summarize import load_summarize_chain

# Load documents
docs = [...]  # Your documents

# Create summarization chain
chain = load_summarize_chain(llm, chain_type="map_reduce")

# Summarize
summary = chain.run(docs)
```

## RAG (Retrieval-Augmented Generation)

**Architecture:**
```
User Query
    ↓
Query Embedding
    ↓
Vector Search (Retrieve relevant documents)
    ↓
Combine Query + Retrieved Context
    ↓
LLM Generation
    ↓
Answer
```

**Implementation:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Use local embeddings (no API needed)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create RAG chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
result = qa_chain({"query": "What are the key findings?"})
```

## Memory Types

### 1. ConversationBufferMemory
Stores entire conversation history
```python
memory = ConversationBufferMemory()
```

### 2. ConversationBufferWindowMemory
Stores last K interactions
```python
memory = ConversationBufferWindowMemory(k=5)
```

### 3. ConversationSummaryMemory
Summarizes old messages
```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
```

### 4. ConversationEntityMemory
Extracts and tracks entities
```python
from langchain.memory import ConversationEntityMemory
memory = ConversationEntityMemory(llm=llm)
```

## Chain Types

### 1. LLMChain
Basic chain with prompt + LLM
```python
chain = LLMChain(llm=llm, prompt=prompt)
```

### 2. SequentialChain
Run multiple chains in sequence
```python
chain = SimpleSequentialChain(chains=[chain1, chain2])
```

### 3. RouterChain
Route to different chains based on input
```python
from langchain.chains.router import MultiPromptChain
chain = MultiPromptChain(...)
```

### 4. TransformChain
Custom transformations
```python
from langchain.chains import TransformChain
chain = TransformChain(transform=custom_function)
```

## Agent Types

| Agent Type | Description | Best For |
|------------|-------------|----------|
| **Zero-Shot ReAct** | Reasons and acts based on tools | General tasks |
| **Conversational** | Maintains conversation memory | Chatbots |
| **Self-Ask** | Asks follow-up questions | Complex reasoning |
| **OpenAI Functions** | Uses function calling | Structured outputs |
| **Plan-and-Execute** | Plans then executes steps | Multi-step tasks |

## Document Loaders

```python
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader
)

# Text file
loader = TextLoader("file.txt")

# PDF
loader = PyPDFLoader("document.pdf")

# Web page
loader = WebBaseLoader("https://example.com")

# Load documents
documents = loader.load()
```

## Vector Stores

| Vector Store | Use Case | Pros |
|--------------|----------|------|
| **FAISS** | Fast local search | Fast, no server needed |
| **Chroma** | Persistent local DB | Easy to use, persists |
| **Pinecone** | Cloud vector DB | Scalable, managed |
| **Weaviate** | Self-hosted | Full-featured, open-source |

## Best Practices

### 1. Prompt Engineering
- Use clear, specific instructions
- Provide examples (few-shot)
- Structure prompts consistently
- Test and iterate

### 2. Cost Optimization
- Cache embeddings
- Use cheaper models for simple tasks
- Implement rate limiting
- Monitor token usage

### 3. Error Handling
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    try:
        result = chain.run(input)
        print(f"Tokens used: {cb.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")
```

### 4. Debugging
```python
# Enable verbose mode
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# Use callbacks
from langchain.callbacks import StdOutCallbackHandler
callbacks = [StdOutCallbackHandler()]
chain.run(input, callbacks=callbacks)
```

## Common Patterns

### Pattern 1: Multi-Document QA
```python
# Load multiple documents
docs = load_documents(["doc1.pdf", "doc2.pdf", "doc3.pdf"])

# Create vector store
vectorstore = create_vectorstore(docs)

# Create QA system
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
```

### Pattern 2: Conversational Agent
```python
# Agent with memory and tools
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory
)
```

### Pattern 3: Custom Chain
```python
from langchain.chains import LLMChain

class CustomChain(LLMChain):
    def _call(self, inputs):
        # Custom preprocessing
        processed = preprocess(inputs)
        # Call LLM
        result = super()._call(processed)
        # Custom postprocessing
        return postprocess(result)
```

## Advanced Topics

- **Custom Tools**: Create domain-specific tools
- **Async Chains**: Non-blocking execution
- **Streaming**: Real-time token generation
- **Callbacks**: Monitor and log execution
- **Output Parsers**: Structure LLM outputs
- **Multi-Modal**: Images, audio, video

## Production Considerations

### Deployment
- Use async for better performance
- Implement caching strategies
- Monitor API costs and usage
- Set up proper error handling
- Use environment variables for secrets

### Scaling
- Connection pooling for databases
- Batch processing for embeddings
- Load balancing for high traffic
- Rate limiting to prevent abuse

## Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)

### Tutorials
- [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart)
- [Build a Chatbot](https://python.langchain.com/docs/use_cases/chatbots)
- [RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [LangChain Twitter](https://twitter.com/langchainai)

## Next Steps

After mastering LangChain:
- **Build Production Apps**: Deploy LangChain applications
- **Chapter 20**: Advanced Concepts (Question Answering)
- **Explore**: LangSmith for monitoring, LangServe for deployment

---

**Build with LangChain!** Create powerful LLM applications with ease!
