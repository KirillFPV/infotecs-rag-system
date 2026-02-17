from qdrant_client import QdrantClient
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings

from systems.VectorDBBuilder import VectorDBBuilder
from systems.LangChainSystem import LangchainRAGSystem

def load_generator(model_name):
    
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"  # Автоматически выбирает подходящий dtype (float16/bfloat16/float32)
    )

    # Настройка паддинг-токена (если не установлен)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Создание пайплайна для генерации текста
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        max_new_tokens=512,
        max_length = None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,
    )

    return qa_pipeline

if __name__=='__main__':

    # Загружаем LLM генератор
    generator_pipeline = load_generator(model_name='Qwen/Qwen3-0.6B')

    # Загружаем embedding модель
    embeddings = HuggingFaceEmbeddings(model_name='Qwen/Qwen3-Embedding-0.6B')

    # Создаем клиента для qdrant
    client = QdrantClient(url="http://localhost:6333")

    # Создаем векторную базу данных
    db_builder = VectorDBBuilder(embedding_model=embeddings)
    vector_store = db_builder.create_qdrant_fromPDF(pdf_path='data', 
                                                    client=client,
                                                    collection_name='infotecs', 
                                                    chunk_size=1000, 
                                                    chunk_overlap=200,
                                                    new_data=True) # Ставим True, чтобы при перезапусках данные не накапливались

    # Инициализиурем RAG систему
    rag_sysytem = LangchainRAGSystem(vector_store=vector_store, llm_pipeline=generator_pipeline, top_k=3)

    while True:
        query = input('Введите ваш вопрос: ')
        answer, docs = rag_sysytem(query)
        print(answer)