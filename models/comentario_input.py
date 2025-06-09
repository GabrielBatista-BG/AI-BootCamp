from pydantic import BaseModel, Field
from typing import List,Optional

class ComentarioInput(BaseModel):
    categoria_principal: Optional[str] = Field(..., alias="Categoria Principal")
    produto: Optional[str] = Field(..., alias="Produto")
    categoria: Optional[str] = Field(..., alias="Categoria")
    subcategoria: Optional[str] = Field(..., alias="Subcategoria")
    titulo_avaliacao: Optional[str] = Field(..., alias="Título da Avaliação")
    avaliacao_geral: int = Field(..., alias="Avaliação Geral")
    recomendaria_a_um_amigo: Optional[str] = Field(..., alias="Recomendaria a um amigo")
    comentario: Optional[str] = Field(..., alias="Comentário")
    id: Optional[str] = Field(..., alias="Id")

    class Config:
        allow_population_by_field_name = True

class ComentariosInput(BaseModel):
    comentarios: List[ComentarioInput]
    
    
    
    