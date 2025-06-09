from typing import Callable
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableSerializable
from dotenv import load_dotenv
from models.comentario_input import ComentariosInput

from config.agents import (
    get_agent_sumarizacao,
    get_agent_gerador_topicos,
    get_agent_chat_rag, 
    get_agent_sentimentos          
)
from config.database import csv_to_sqlite
from config.model import load_model
from retrievers import (
    product_brand_retriever,
    product_name_retriever,
    site_category_lv1_retriever,
    site_category_lv2_retriever,
)


def get_app() -> FastAPI:
    app = FastAPI(
        docs_url="/docs",
        redoc_url="/redoc",
        root_path="/api",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8081"],
        allow_origin_regex=r"http://localhost(:[0-9]+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.models = {'name': 'nome_do_modelo'}

    def start_app_handler() -> Callable:

        def startup() -> None:
            csv_filepath = rf'./src/B2W-Reviews01.csv'
            csv_to_sqlite(csv_filepath)


            class Consts:
                model: RunnableSerializable[dict, str] = None
                chat_agent = None

            app.consts = Consts()
            app.consts.model = load_model("mistral")

            pass

        return startup

    app.add_event_handler("startup", start_app_handler())

    return app


app = get_app()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/model")
async def model():
    return {"message": str(app.consts.model)}


@app.get("/product_brand/{product_brand}")
async def brands(product_brand: str):
    return product_brand_retriever(product_brand)


@app.get("/product_name/{product_name}")
async def product_name(product_name: str):
    return product_name_retriever(product_name)


@app.get("/site_category_lv1/{site_category_lv1}")
async def site_category_lv1(site_category_lv1: str):
    return site_category_lv1_retriever(site_category_lv1)


@app.get("/site_category_lv2/{site_category_lv2}")
async def site_category_lv2(site_category_lv2: str):
    return site_category_lv2_retriever(site_category_lv2)


# @app.get("/search/{search_type}/{search_query}")
# async def search_and_summarize(search_type: str, search_query: str):
#     match search_type:
#         case 'product_brand':
#             retriever_result = product_brand_retriever(search_query)
#         case 'product_name':
#             retriever_result = product_name_retriever(search_query)
#         case 'site_category_lv1':
#             retriever_result = site_category_lv1_retriever(search_query)
#         case 'site_category_lv2':
#             retriever_result = site_category_lv2_retriever(search_query)
#         case _:
#             return {"status": "error", "message": "search_type inválido"}

#     model = app.consts.model
#     agent_gerador_topicos = get_agent_gerador_topicos(model)
#     tentativa = 0
#     while True:
#         try:
#             topicos = agent_gerador_topicos.invoke(retriever_result)
#             # agent_sumarizacao = get_agent_sumarizacao(model, topicos)
#             # sumarizacao = agent_sumarizacao.invoke({"query": retriever_result})

#             return {
#                 'status': 'ok',
#                 'sumarizacao': "sumarizacao",
#                 'topicos': topicos,
#                 'retriever_result': retriever_result,
#             }
#         except Exception as e:
#             print(f"Erro ao tentar sumarizar (tentativa {tentativa}): {e}")
#             tentativa += 1
#             if tentativa >= 5:
#                 return {
#                     'status': 'error',
#                     'sumarizacao': None,
#                     'topicos': None,
#                     'retriever_result': None,
#                 }
            
@app.post("/sentimentos")
async def sentimentos(dados: ComentariosInput):
    list_comentarios = dados.comentarios
    model = app.consts.model
    agent_sentimentos = get_agent_sentimentos(model)
    listResult = []
    for comment in list_comentarios:
        sentimentos = agent_sentimentos.invoke({"query":comment})
        listResult.append(sentimentos)
    return {
        "sentimentos":listResult
    }  

@app.post("/gerador_topicos")
async def gerador_topicos(dados: ComentariosInput):
    list_comentarios = dados.comentarios
    model = app.consts.model
    agent_gerador_topicos = get_agent_gerador_topicos(model)
    listResult = []
    for comentario in list_comentarios:
        obj_comentario_topico = {}
        topicos = agent_gerador_topicos.invoke({"query":comentario})
        obj_comentario_topico["comentario"] = comentario
        obj_comentario_topico["topicos_principais"] = topicos
        listResult.append(obj_comentario_topico)
    return {
        "result":listResult
    }

@app.post("/sumarizacao")
async def sumarizador(dados: ComentariosInput):
    list_comentarios = dados.comentarios
    model = app.consts.model
    agent_sumarizacao = get_agent_sumarizacao(model)
    result = agent_sumarizacao.invoke({"query":list_comentarios})
    return result



@app.post("/chat")
async def chat(message: str):
    """
    Este endpoint agora invoca o agente de chat RAG que foi instanciado em startup.
    Ele possui:
      - Memória de conversa (ConversationBufferMemory),
      - Ferramentas FAISS (@tool),
    Basta chamar `.run(message)` que o agente cuidará de:
      1) Recuperação via FAISS (caso julgue necessário),
      2) Incluir histórico no prompt,
      3) Gerar resposta com base no modelo de AI
      4) Armazenar tudo em memória para chamadas futuras.
    """
    # Chama o agente RAG que foi criado em startup
    model = app.consts.model
    agent_chat = get_agent_chat_rag(model)
    config = {"configurable": {"thread_id": "1"}}
    response = agent_chat.invoke({"input": message},config=config)
    return {"resposta": response}



from fastapi import HTTPException

@app.get("/sentimento_geral/{search}/{query}/{qtd_comentario}")
async def sentimento_geral(search: str, query: str, qtd_comentario: str):
    """
    Esse endpoint invoca o agente de sentimento geral que busca os comentários
    e devolve um objeto SentimentosModel como dict.
    """
    try:
        qtd_comentario = int(qtd_comentario)
    except ValueError:
        raise HTTPException(status_code=400, detail="qtd_comentario deve ser um número inteiro.")

    if qtd_comentario <= 0:
        raise HTTPException(status_code=400, detail="qtd_comentario deve ser maior que zero.")

    match search:
        case 'product_brand':
            retriever_result = product_brand_retriever(query, qtd_comentario)
        case 'product_name':
            retriever_result = product_name_retriever(query, qtd_comentario)
        case 'site_category_lv1':
            retriever_result = site_category_lv1_retriever(query, qtd_comentario)
        case 'site_category_lv2':
            retriever_result = site_category_lv2_retriever(query, qtd_comentario)
        case _:
            raise HTTPException(status_code=400, detail="Parâmetro 'search' inválido.")

    model = app.consts.model
    agent_sumarizacao = get_agent_sumarizacao(model)
    result = agent_sumarizacao.invoke({"query": retriever_result})

    return result



if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI + Ollama (Mistral) + FAISS RAG")
    uvicorn.run("main:app", port=8081, reload=True)
