import uuid
from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document

class VectorStoreManager:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name="multi_modal_rag",
            embedding_function=OpenAIEmbeddings()
        )
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            id_key=self.id_key
        )

    def add_documents(self, documents: List[Any], summaries: List[str]) -> None:
        """Add documents and their summaries to the vector store."""
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        summary_docs = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        
        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, documents)))

    def add_images(self, images, summaries: List[str]) -> None:
        """Add images and their summaries to the vector store."""
        img_ids = [str(uuid.uuid4()) for _ in images]
        
        summary_imgs = [
            Document(page_content=summary, metadata={self.id_key: img_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        
        self.retriever.vectorstore.add_documents(summary_imgs)
        self.retriever.docstore.mset(list(zip(img_ids, images)))
