
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.helpers import DocumentParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage


class RAGPipeline:
    def __init__(self, retriever, model_name="gpt-4o-mini"):
        self.retriever = retriever
        self.model_name = model_name
        
    def _build_prompt(self, kwargs: Dict[str, Any]) -> ChatPromptTemplate:
        """Build the prompt for the RAG pipeline."""
        docs_by_type = kwargs['context']
        user_question = kwargs['question']

        context_text = " ".join(text_element.text for text_element in docs_by_type["texts"])

        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text}
        Question: {user_question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]
        
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })
        
        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def create_chain(self):
        """Create the RAG chain."""
        return (
            {
                "context": self.retriever | RunnableLambda(DocumentParser.parse_documents),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._build_prompt)
            | ChatOpenAI(model=self.model_name)
            | StrOutputParser()
        )

    def create_source_chain(self):
        """Create the RAG chain with sources."""
        return (
            {
                "context": self.retriever | RunnableLambda(DocumentParser.parse_documents),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(self._build_prompt)
                    | ChatOpenAI(model=self.model_name)
                    | StrOutputParser()
                )
            )
        )
