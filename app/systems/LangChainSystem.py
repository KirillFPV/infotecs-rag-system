from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
import re

class LangchainRAGSystem:
    """
    RAG система на базе LangChain с Qdrant векторным хранилищем.
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        llm_pipeline: pipeline,
        top_k: int = 5
    ):
        """
        Args:
            vector_store: Qdrant векторное хранилище
            llm_pipeline: Pipeline для генерации ответов
            top_k: Количество релевантных документов для поиска
        """
        self.vector_store = vector_store
        self.llm_pipeline = llm_pipeline
        self.top_k = top_k
        
        # Создание ретривера
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        # Настройка промпта
        self.prompt_template = """Ты консультант по продукции InfoTecs. Тебе необходимо ответить на вопрос пользователя, используя данный тебе контекст. Ответ должен быть объёмным и закрыть все вопросы пользователя.

ВАЖНЫЕ ПРАВИЛА ФОРМАТИРОВАНИЯ:
1. Начинай свой ответ строго с "ANSWER:"
2. Заканчивай свой ответ строго с "[END ANSWER]"
3. Ничего не пиши до ANSWER: и после [END ANSWER]

Контекст:
{context}

Вопрос: {question}

Ответ:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Создание LangChain LLM обертки
        self.langchain_llm = HuggingFacePipeline(pipeline=llm_pipeline)
        
        # Создание QA цепочки
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.langchain_llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )
    
    def _postprocess_answer(self, result: Dict[str, Any]) -> str:
        """Извлечение ответа из результата."""
        answer = result['result']
        
        # Поиск текста между ANSWER: и [END ANSWER]
        pattern = r'ANSWER:\s*(.*?)\s*\[END ANSWER\]'
        match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
        if match:
            return f"ANSWER: {match.group(1).strip()} [END ANSWER]"
        
        print(f'[DEBUG] Full answer: {result['result']}')
        
        return answer
    
    def query(self, user_input: str) -> Tuple[str, List[str]]:
        """
        Основной метод для обработки запроса.
        
        Returns:
            Tuple[answer, source_documents]
        """
        result = self.qa_chain.invoke({"query": user_input})
        answer = self._postprocess_answer(result)
        source_docs = [doc.page_content for doc in result['source_documents']]
        return answer, source_docs
    
    def __call__(self, user_input: str) -> Tuple[str, List[str]]:
        """Позволяет вызывать объект как функцию."""
        return self.query(user_input)