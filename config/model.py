from langchain_ollama import OllamaEmbeddings, ChatOllama


def load_model(model_name) -> ChatOllama:
    model = ChatOllama(model=model_name, name=model_name)

    chain = model

    return chain

def load_embedding_model() :
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings
