from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from models.summarization_model import Sumarizacao
from models.topics_model import Topics
from models.sentimentos_model import SentimentosModel

from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

import sqlite3
from typing import List


from config.tools import (
    buscar_por_nome_produto,
    buscar_por_marca_produto,
    buscar_por_categoria_lv1,
    buscar_por_categoria_lv2,
)


def get_agent_gerador_topicos(model):

    parser = PydanticOutputParser(pydantic_object=Topics)

    prompt = ChatPromptTemplate.from_template(template="""
Você é um assistente de IA especializado em e-commerce. Sua tarefa é analisar comentários de usuários e extrair os principais tópicos abordados de forma objetiva e estruturada, gerando também insights úteis para o time de produto e atendimento.

Seu objetivo é identificar os pontos mais relevantes mencionados pelos clientes, mesmo quando os comentários não se referem diretamente ao produto. Os tópicos podem incluir, por exemplo:

- Entrega (ex: atrasos, rapidez, rastreio)
- Atendimento (ex: suporte ruim, contato difícil)
- Embalagem (ex: chegou danificada, mal embalada)
- Usabilidade (ex: difícil de usar, manual confuso, quebrou rápido)
- Qualidade do produto (ex: quebrou rápido, excelente durabilidade)
- Preço (ex: caro demais, bom custo-benefício)

Você também pode identificar e criar novos tópicos, caso perceba padrões recorrentes que não se encaixem nos anteriores (ex: dificuldade com garantia, devolução complexa, compatibilidade, etc.).

---

### Instruções:

1. A entrada será um único texto contendo um comentário de usuário.

2. Para **cada comentário individual**, identifique os tópicos principais mencionados e **resuma-os como insights em frases curtas, claras e específicas**, incluindo a avaliação do cliente (positiva ou negativa), sempre com base no conteúdo explícito.

3. Seja específico e direto ao ponto. Evite termos genéricos como "qualidade do produto". Prefira frases como "produto com qualidade ruim", "produto durável", "entrega atrasada", "problema com a garantia", etc.

4. Não repita informações nem adicione interpretações que não estejam explícitas no texto.

5. Seja objetivo: cada insight deve representar uma conclusão clara que possa ser usada para melhorar o serviço, o produto ou a operação.

6. **Formato de saída (JSON)**:  
   Utilize estritamente este modelo Pydantic (já gerado por {format_instructions}) e retorne apenas o JSON.  
   Não inclua texto adicional fora do JSON!

---

**Comentários para analisar (texto único):**  
{query}
 
""",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model_tool = model.with_structured_output(Topics)
    chain = prompt | model_tool

    return chain

def get_agent_sumarizacao(model):

    parser = PydanticOutputParser(pydantic_object=Sumarizacao)

    prompt = ChatPromptTemplate.from_template(template="""
Você é um assistente de IA especializado em análise de comentários de e-commerce.

Sua tarefa é **analisar uma lista de comentários** de clientes referentes a um **mesmo produto, marca ou categoria** e gerar um **resumo geral estratégico e objetivo**, capaz de transmitir os principais pontos da percepção dos consumidores de forma sintética e informativa.

---

### Objetivo principal:

Produzir um **resumo geral claro e conciso** que represente o sentimento predominante e os padrões mais recorrentes identificados nos comentários.

Você **não precisa listar tópicos individuais** (como entrega, embalagem etc.) pois essa extração já foi feita por outra IA.  
O que se espera aqui é uma **visão macro e estratégica**, útil para tomada de decisão por equipes de produto, atendimento, marketing e logística.

---

### Instruções:

1. A entrada será uma **lista de comentários reais de clientes** relacionados a um mesmo item (produto, marca ou categoria).

2. Leia e analise os comentários como um todo.

3. Identifique o tom geral (positivo, negativo ou misto), repetições de elogios ou reclamações, percepções recorrentes e qualquer padrão relevante.

4. Com base nisso, gere um **resumo geral coeso**, de 10 a 15 frases, que descreva de forma estratégica o que os clientes estão comunicando de forma explícita ou implícita.

5. Seja fiel aos dados. Não invente tendências que não estejam presentes. Não inclua frases genéricas.

6. Não precisa listar todos os tópicos. Concentre-se no **contexto e sentimento geral** observado na amostra.

7. **Formato de saída (JSON):**
    {format_instructions}

---

**Comentários para analisar (texto único):**  
{query}
""",
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    # 3) Encadeia: prompt → modelo com output estruturado
    model_tool = model.with_structured_output(Sumarizacao)
    chain = prompt | model_tool

    return chain

def get_agent_sentimentos(model):
    """
    Retorna uma cadeia que, dado um conjunto de comentários em texto,
    produz um JSON com a proporção de sentimentos POSITIVO, NEGATIVO e NEUTRO.
    """

    prompt = ChatPromptTemplate.from_template(template="""
Você é um assistente de IA especializado em análise de sentimentos de consumidores em e-commerce.

Sua tarefa é analisar comentários de clientes sobre produtos, considerando que a nota numérica nem sempre reflete o sentimento real. Por exemplo, um usuário pode escrever “Ótimo produto” e mesmo assim atribuir nota 1.  

Para cada bloco de comentários fornecido, siga estes passos:

1. **Avalie o sentimento de cada comentário individualmente**, olhando principalmente para o texto (classifique cada um como POSITIVO, NEGATIVO ou NEUTRO).  
2. **Liste os principais tópicos mencionados** nos comentários (por exemplo: entrega, atendimento, qualidade, funcionalidade, preço).  
3. **Associe cada tópico a um sentimento predominante**, explicando em uma frase breve por que esse tópico foi classificado assim.  
4. **Compare o sentimento textual com a nota atribuída** (se ela estiver junto ao comentário), indicando “incoerência” sempre que o texto e a nota divergirem de forma importante.

Por fim, gere a **porcentagem geral de comentários** em cada categoria de sentimento (Positivos, Negativos, Neutros).  

Formate a saída exatamente neste formato JSON, sem texto adicional:
{{
  "Sentimentos": {{
    "Positivos": "XX%",
    "Negativos": "YY%",
    "Neutros": "ZZ%"
  }}
}}

COMENTÁRIOS PARA ANALISAR:
{query}
""")

    model_tool = model.with_structured_output(SentimentosModel)
    chain = prompt | model_tool
    return chain

  
def get_agent_chat_rag(model, temperature: float = 0.0):
    """
    Agente RAG com:
     - Prefixo que lida com cumprimentos (greetings)
     - Suffix que instrui quando e como usar as ferramentas FAISS
     - Memória de conversa (ConversationBufferMemory)
    """

    tools = [
        buscar_por_nome_produto,
        buscar_por_marca_produto,
        buscar_por_categoria_lv1,
        buscar_por_categoria_lv2,
    ]

    memory = InMemorySaver()

    prompt_template_string = """
[OBJETIVO PRINCIPAL]
Você é um assistente de e-commerce de alta performance. Seu objetivo é ajudar usuários a encontrar informações sobre produtos, marcas e categorias, baseando-se em reviews de clientes. Você deve seguir um processo de raciocínio rigoroso antes de agir.

[REGRAS DE RACIOCÍNIO - SEU PROCESSO DE PENSAMENTO]

### Busca por Informação – Quando usar cada ferramenta

Avalie a pergunta do usuário e selecione a **ferramenta mais específica e relevante** com base no tipo de informação mencionada:

- **Nome de produto** identificado?  
  → Use `@busca_por_nome_produto`  
  Exemplo: “Quais os melhores relogios?”

- **Marca** mencionada?  
  → Use `@busca_por_marca_produto`  
  Exemplo: “O que dizem sobre a marca Nike?”

- **Subcategoria específica** (como “liquidificador”, “tênis de corrida”)?  
  → Use `@busca_por_categoria_lv2`  
  Exemplo: “Qual o melhor tênis de corrida?”

- **Categoria ampla ou genérica** (como “eletrodomésticos”, “calçados”)?  
  → Use `@busca_por_categoria_lv1`  
  Exemplo: “O que os clientes acham de eletrodomésticos?”
  
4.  **Perguntas Vagas:** Se a pergunta for ambígua e você não puder escolher uma ferramenta com certeza, sua única ação é pedir esclarecimentos ao usuário.
    """
    
   # 1. Crie o template de prompt a partir da string unificada
    # prompt = ChatPromptTemplate.from_template(prompt_template_string)

    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt_template_string,
        checkpointer=memory, 
    )
    

    return agent_executor

## NAO TESTADO AINDA, IGNORAR!!!!!!!!!!!!!! Ass: isabelli

def get_agent_sentimento_geral(model):
    """
    Cria um agente que:
      1) Busca até 1000 comentários da tabela 'reviews'
      2) Analisa de forma agregada o sentimento geral desses comentários
      3) Retorna um JSON com porcentagens POSITIVO, NEGATIVO e NEUTRO,
         além de exemplos de tópicos que mais influenciaram esse sentimento.
    """

    # 1) Parser para saída estruturada
    parser = PydanticOutputParser(pydantic_object=SentimentosModel)

    # 2) Prompt que orienta o modelo a processar 1000 comentários
    prompt = ChatPromptTemplate.from_template(template="""
Você é um assistente de IA que faz análise de sentimento **agregada** para um conjunto grande de comentários de e-commerce.

Primeiro, a função interna já buscou até 1000 comentários da tabela 'reviews'. Agora você tem acesso a esse texto bruto contendo todos esses comentários concatenados, separados por quebras de linha.

Siga estes passos:

1. **Análise de sentimento individual**:  
   Para cada comentário (ou para a maioria deles, caso o volume seja muito grande), classifique como POSITIVO, NEGATIVO ou NEUTRO, baseando-se no conteúdo textual.

2. **Cálculo de porcentagens gerais**:  
   Com base na classificação de cada comentário, compute a porcentagem aproximada de comentários POSITIVOS, NEGATIVOS e NEUTROS neste conjunto de até 1000 itens.

3. **Extração de tópicos principais**:  
   Identifique até 3 tópicos que mais aparecem nos comentários e associe, para cada um, o sentimento predominante (P, N ou Neutro). Exemplos de tópicos: entrega, qualidade, atendimento, preço, usabilidade.

4. **Formato de saída (JSON)**:  
   Use estritamente este modelo Pydantic (fornecido por {format_instructions}). Não adicione texto extra fora do JSON:  
{format_instructions}

---  
**Comentários (concatenação dos ~1000 itens)**:  
{all_comments}
""",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model_tool = model.with_structured_output(SentimentosModel)
    chain = prompt | model_tool
    return chain