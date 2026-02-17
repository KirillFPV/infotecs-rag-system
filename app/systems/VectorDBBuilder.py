import faiss
import numpy as np
from typing import List, Optional, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from systems.PDFPlumberLoader import PDFPlumberLoader


class VectorDBBuilder:
    """
    Класс для создания векторных хранилищ на основе FAISS и Qdrant.
    
    Параметры
    ----------
    embedding_model : объект с методами embed_documents и embed_query
        Модель эмбеддингов, совместимая с интерфейсом LangChain.
        Пример: HuggingFaceEmbeddings(model_name="...")
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def create_faiss(
        self,
        texts: List[str],
        index_path: str,
        texts_path: str
    ) -> Tuple[faiss.Index, List[str]]:
        """
        Создаёт FAISS-индекс и сохраняет его вместе с текстами.

        Аргументы
        ---------
        texts : list of str
            Список текстовых фрагментов (чанков).
        index_path : str
            Путь для сохранения файла индекса FAISS.
        texts_path : str
            Путь для сохранения текстов (по одному чанку на строку).

        Возвращает
        ----------
        index : faiss.Index
            Построенный индекс.
        texts : list of str
            Исходные тексты (для дальнейшего использования).
        """
        # Получаем векторы для всех текстов
        vectors = self.embedding_model.embed_documents(texts)
        vectors = np.array(vectors).astype('float32')

        # Создаём и заполняем индекс
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, index_path)

        # Сохраняем тексты
        with open(texts_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        return index, texts

    def create_qdrant(
        self,
        texts: List[str],
        client: QdrantClient,
        collection_name: str,
        metadata: Optional[List[dict]] = None
    ) -> QdrantVectorStore:
        """
        Создаёт коллекцию в Qdrant и загружает туда документы.

        Аргументы
        ---------
        texts : list of str
            Список текстовых фрагментов.
        client : QdrantClient
            Инициализированный клиент Qdrant (например, с указанием пути).
        collection_name : str
            Имя коллекции.
        metadata : list of dict, optional
            Список словарей с метаданными для каждого текста (должен совпадать по длине с texts).

        Возвращает
        ----------
        vector_store : QdrantVectorStore
            Объект векторного хранилища, готовый к использованию в цепочках LangChain.
        """
        # Создаём документы LangChain (с метаданными, если они переданы)
        if metadata is not None:
            if len(metadata) != len(texts):
                raise ValueError("Длина metadata должна совпадать с длиной texts")
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadata)
            ]
        else:
            documents = [Document(page_content=text) for text in texts]

        # Определяем размерность вектора (через один запрос)
        vector_size = len(self.embedding_model.embed_query("sample"))

        # Создаём коллекцию, если её ещё нет
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # можно также использовать Distance.EUCLIDEAN
                )
            )

        # Инициализируем хранилище и добавляем документы
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embedding_model,
        )
        vector_store.add_documents(documents)

        return vector_store

    def create_qdrant_fromPDF(
        self,
        pdf_path: str,
        client: QdrantClient,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
        glob_pattern: str = "**/*.pdf",
        new_data: bool = False  # Новый флаг для принудительной перезагрузки
    ) -> QdrantVectorStore:
        """
        Загружает PDF-документы из папки, разбивает на чанки и добавляет в коллекцию Qdrant.
        Если коллекция уже существует и new_data=False, загрузка пропускается.
        Если new_data=True, существующая коллекция удаляется и создаётся заново.

        Аргументы
        ---------
        pdf_path : str
            Путь к папке, содержащей PDF-файлы.
        client : QdrantClient
            Инициализированный клиент Qdrant.
        collection_name : str
            Имя коллекции в Qdrant.
        chunk_size : int
            Размер чанка в символах.
        chunk_overlap : int
            Перекрытие между соседними чанками.
        glob_pattern : str, optional
            Шаблон для поиска PDF-файлов (по умолчанию "**/*.pdf").
        new_data : bool, optional
            Если True, игнорирует наличие коллекции и принудительно перезаписывает данные.

        Возвращает
        ----------
        vector_store : QdrantVectorStore
            Объект векторного хранилища, содержащий загруженные документы.
        """
        # Проверяем существование коллекции
        collection_exists = client.collection_exists(collection_name)

        # Если new_data=False и коллекция уже есть – просто подключаемся к ней
        if not new_data and collection_exists:
            print(f"Коллекция '{collection_name}' уже существует. Загрузка пропущена. "
                  "Используйте new_data=True для принудительной перезагрузки.")
            return self.load_qdrant(client, collection_name, self.embedding_model)

        # Если new_data=True и коллекция существует – удаляем её для последующей перезаписи
        if new_data and collection_exists:
            print(f"Удаление существующей коллекции '{collection_name}' для перезагрузки.")
            client.delete_collection(collection_name)

        # --- Загрузка и обработка PDF ---
        # 1. Загрузка всех PDF-файлов из указанной папки
        loader = DirectoryLoader(
            path=pdf_path,
            glob=glob_pattern,
            loader_cls=PDFPlumberLoader,
        )
        documents = loader.load()
        print(f"Загружено документов: {len(documents)}")

        # 2. Разбиение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Получено фрагментов: {len(chunks)}")

        # 3. Извлечение текстов и метаданных из чанков
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        # 4. Использование существующего метода create_qdrant для загрузки
        vector_store = self.create_qdrant(
            texts=texts,
            client=client,
            collection_name=collection_name,
            metadata=metadata
        )
        print(f"Данные успешно загружены в коллекцию '{collection_name}'")

        return vector_store

    @staticmethod
    def load_faiss(index_path: str, texts_path: str) -> Tuple[faiss.Index, List[str]]:
        """
        Загружает ранее сохранённый FAISS-индекс и соответствующие тексты.

        Аргументы
        ---------
        index_path : str
            Путь к файлу индекса FAISS.
        texts_path : str
            Путь к файлу с текстами.

        Возвращает
        ----------
        index : faiss.Index
            Загруженный индекс.
        texts : list of str
            Список текстов.
        """
        index = faiss.read_index(index_path)
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = [line.rstrip('\n') for line in f]
        return index, texts

    @staticmethod
    def load_qdrant(
        client: QdrantClient,
        collection_name: str,
        embedding_model: Any
    ) -> QdrantVectorStore:
        """
        Подключается к существующей коллекции Qdrant.

        Аргументы
        ---------
        client : QdrantClient
            Клиент Qdrant.
        collection_name : str
            Имя коллекции.
        embedding_model : объект с методами embed_query/embed_documents
            Модель эмбеддингов, которая использовалась при создании коллекции.

        Возвращает
        ----------
        vector_store : QdrantVectorStore
            Объект векторного хранилища.
        """
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )