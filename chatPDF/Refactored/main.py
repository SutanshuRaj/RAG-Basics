from config.settings import Settings
from core.document_processor import DocumentProcessor
from core.rag_pipeline import RAGPipeline
from model.vector_store import VectorStoreManager
from utils.helpers import ImageUtils
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import timeit

def main():
    # Initialize components
    settings = Settings()
    doc_processor = DocumentProcessor(settings)
    vector_store = VectorStoreManager()
    
    # Process PDF
    processed_docs = doc_processor.process_pdf('Attention.pdf')
    images = ImageUtils.extract_base64_images(processed_docs["chunks"])
    
    # Create summarization chain
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary, no additional comments.
    Do not start your message by saying "Here is a summary."
    Just give the summary as it is. 
    Table or text chunk: {element}
    """
    
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summary_chain = (
        {"element": lambda x: x}
        | ChatPromptTemplate.from_template(prompt_text)
        | model
        | StrOutputParser()
    )
    
    # Generate summaries
    text_summaries = summary_chain.batch(processed_docs["texts"], {"max_concurrency": 3})
    table_summaries = summary_chain.batch(
        [table.metadata.text_as_html for table in processed_docs["tables"]],
        {"max_concurrency": 3}
    )
    
    # Add to vector store
    vector_store.add_documents(processed_docs["texts"], text_summaries)
    # vector_store.add_documents(images, image_summaries)
    
    # Create and use RAG pipeline
    rag = RAGPipeline(vector_store.retriever)
    chain = rag.create_chain()
    
    start = timeit.default_timer()
    response = chain.invoke("What is the Attention mechanism?")
    stop = timeit.default_timer()
    
    print(f"Time taken: {round(stop-start, 2)} seconds")
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()