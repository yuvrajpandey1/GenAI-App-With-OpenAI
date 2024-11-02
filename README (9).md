
# ChatBot using Ollama

Assistant ChatBot to answers normal query and any kind of information by using Ollama.Ollama is a framework that allows you to integrate large language models like Llama into various applications, including chatbots.

We can use this model at different places like:

- Customer Support: It can help answer frequently asked questions by pulling information directly from product manuals, FAQs, and other support documents.

- Educational Tools: Students and educators can use it to get quick answers from textbooks, research papers, and other academic resources.



# Documentation 

[OpenAIEmbeddings](https://js.langchain.com/v0.2/docs/integrations/text_embedding/)

Embedding models create a vector representation of a piece of text.This page documents integrations with various model providers that allow you to use embeddings in LangChain.

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it.You can embed queries for search with embedQuery. This generates a vector representation specific to the query

[FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

Facebook AI Similarity Search (FAISS) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also includes supporting code for evaluation and parameter tuning.

Faiss is written in C++ with complete wrappers for Python. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at FAIR, the fundamental AI research team of Meta.


[Vector stores](https://levelup.gitconnected.com/creating-retrieval-chain-with-langchain-f359261e0b85)

One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search for you.


```javascript
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('../../../state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
```

























 








## Important Libraries Used

 - [WebBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base/)
 - [Document loaders](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/)
- [RunnableWithMessageHistory](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/)
 - [Text Splitters](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
 - [PromptTemplate](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/)
- [OpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/)

- [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)

- [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/)






## Plateform or Providers

 - [langchain_groq](https://python.langchain.com/docs/integrations/chat/groq/)
 - [https://huggingface.co/blog/langchain](https://smith.langchain.com/hub)

## Model

 - LLM - gpt-4o


## Installation

Install below libraries

```bash
  pip install langchain
  pip install langchain_community
  pip install langchain-core
  pip install langchain_openai
  pip install OpenAI
  pip install bs4


```
    
## Tech Stack

**Client:** Python, LangChain PromptTemplate, OpenAI,OllamaEmbeddings,FAISS

**Server:** Anaconda Navigator, Jupyter Notebook


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`LANGCHAIN_API_KEY`
`OPENAI_API_KEY`



## Required Libraries 

To install langchain

```bash
  pip install
```

