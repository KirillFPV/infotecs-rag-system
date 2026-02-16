from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re

class BaselineRAGSystem:
    """
    Базовая RAG система с FAISS индексом и кастомной реализацией.
    """
    
    def __init__(
        self,
        vector_store: Tuple[faiss.Index, List[str]],  # (faiss_index, texts)
        embedding_model: SentenceTransformer,
        llm_pipeline: pipeline,
        top_k: int = 5,
        max_context_size: int = 1000
    ):
        """
        Args:
            vector_store: Кортеж (FAISS индекс, список текстов)
            embedding_model: Модель для создания эмбеддингов
            llm_pipeline: Pipeline для генерации ответов
            top_k: Количество релевантных чанков для поиска
            max_context_size: Максимальный размер контекста в символах
        """
        self.index, self.texts = vector_store
        self.embedding_model = embedding_model
        self.llm_pipeline = llm_pipeline
        self.top_k = top_k
        self.max_context_size = max_context_size
        
        # Системный промпт для генерации ответов
        self.system_prompt = """Ты консультант по продукции InfoTecs. Тебе необходимо ответить на вопрос пользователя, используя данный тебе контекст. Ответ должен быть объёмным и закрыть все вопросы пользователя.

ВАЖНЫЕ ПРАВИЛА ФОРМАТИРОВАНИЯ:
1. Начинай свой ответ строго с "ANSWER:"
2. Заканчивай свой ответ строго с "[END ANSWER]"
3. Ничего не пиши до ANSWER: и после [END ANSWER]

Пример правильного ответа:
ANSWER: Ваш развернутый ответ на вопрос пользователя здесь... [END ANSWER]"""
    
    def search_relevant_chunks(self, query: str) -> List[str]:
        """Поиск релевантных чанков по запросу."""
        query_vector = self.embedding_model.encode([query])[0].astype('float32')
        query_vector = np.expand_dims(query_vector, axis=0)
        
        distances, indices = self.index.search(query_vector, self.top_k)
        relevant_chunks = [self.texts[i] for i in indices[0]]
        return relevant_chunks
    
    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        """Генерация ответа на основе релевантных чанков."""
        if not relevant_chunks:
            return "ANSWER: Не удалось найти релевантные фрагменты. [END ANSWER]"
        
        context = " ".join(relevant_chunks)[:self.max_context_size]
        if not context.strip():
            return "ANSWER: Контекст пуст. Невозможно сгенерировать ответ. [END ANSWER]"
        
        input_text = f"{self.system_prompt}\n\nВопрос: {query}\nКонтекст: {context}\n:"
        generated_output = self.llm_pipeline(input_text)[0]['generated_text']
        
        return self._postprocess_answer(generated_output)
    
    def _postprocess_answer(self, generated_output: str) -> str:
        """Очистка сгенерированного ответа."""
        if not generated_output or not isinstance(generated_output, str):
            return ""
        
        # Поиск текста между ANSWER: и [END ANSWER]
        pattern = r'ANSWER:\s*(.*?)\s*\[END ANSWER\]'
        match = re.search(pattern, generated_output, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if answer:
                return f"ANSWER: {answer} [END ANSWER]"
        
        # Fallback: возвращаем как есть
        return generated_output.strip()
    
    def query(self, user_input: str) -> Tuple[str, List[str]]:
        """
        Основной метод для обработки запроса.
        
        Returns:
            Tuple[answer, relevant_chunks]
        """
        relevant_chunks = self.search_relevant_chunks(user_input)
        answer = self.generate_answer(user_input, relevant_chunks)
        return answer, relevant_chunks
    
    def __call__(self, user_input: str) -> Tuple[str, List[str]]:
        """Позволяет вызывать объект как функцию."""
        return self.query(user_input)