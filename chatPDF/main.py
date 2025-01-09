### Reference: Alejandro AO - Multimodal RAG.

import os
import io
import uuid
import base64
import timeit
import PIL.Image

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from IPython.display import Image

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever


load_dotenv()


def imageBase64(images):
	image_b64 = []
	for img in images:
		if "CompositeElement" in str(type(img)):
			image_meta = img.metadata.orig_elements
			for ele in image_meta:
				if "Image" in str(type(ele)):
					image_b64.append(ele.metadata.image_base64)

	return image_b64


def displayBase64(codeBase64):
	img_object = base64.b64decode(codeBase64)
	img = Image(data=img_object)
	img_data = PIL.Image.open(io.BytesIO(img.data))
	img_data.show()


def parseDocs(docs):
	b64 = []
	text = []
	for doc in docs:
		try:
			base64.b64decode(doc)
			b64.append(doc)
		except Exception as e:
			text.append(doc)
	return {'images': b64, 'texts': text}


def buildPrompt(kwargs):
	docs_by_type = kwargs['context']
	user_question = kwargs['question']

	context_text = " "
	if len(docs_by_type["texts"]) > 0:
		for text_element in docs_by_type["texts"]:
			context_text += text_element.text

	prompt_template = f"""
    	Answer the question based only on the following context, which can include text, tables, and the below image.
    	Context: {context_text}
    	Question: {user_question}
		"""

	prompt_content = [{"type": "text", "text": prompt_template}]
	if len(docs_by_type["images"]) > 0:
		for image in docs_by_type["images"]:
			prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
	
	return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])



### Step 1: Extract the Data ###

# LangChain has Loader for Unstructured.
output_path = "./docs/"
file_path = output_path + 'Attention.pdf'

chunks = partition_pdf(
	filename=file_path,
	strategy="hi_res",
	infer_table_structure=True,

	extract_image_block_types=["Image", "Table"],
	# extract_image_block_output_dir=output_path,

	# base64 Representation of Image for the API.
	extract_image_block_to_payload=True,

	chunking_strategy="by_title",
	max_characters=7000,
	combine_text_under_n_chars=1000,
	new_after_n_chars=3000)

print(set([str(type(ele)) for ele in chunks]))
# Each CompositeElement containes related elements, to use together in a RAG pipeline.
# print(chunks[0].metadata.orig_elements)

tables = []
texts = []

for chunk in chunks:
	if 'Table' in str(type(chunk)):
		tables.append(chunk)

	if 'CompositeElement' in str(type(chunk)):
		texts.append(chunk)

images = imageBase64(chunks)

if False:
	displayBase64(images[0])


### Step 2: Summarize the Data ###

# Vectorize the Data, to be used in the Retrieval Process.
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comments.
Do not start your message by saying "Here is a summary."
Just give the summary as it is. 

Table or text chunk: {element}
"""

prompt = ChatPromptTemplate.from_template(prompt_text)

# Chain of Thought.
model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
summaryChain = {"element": lambda x: x} | prompt | model | StrOutputParser()


summarizeText = summaryChain.batch(texts, {"max_concurrency": 3})

# Visualize: table[0].to_dict()
tableHTML = [table.metadata.text_as_html for table in tables]
summarizeTable = summaryChain.batch(tableHTML, {"max_concurrency": 3})

# Quick Sanity Check.
print(summarizeText[0])

prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers architecture. 
                  Be specific about graphs, such as bar plots."""

messages = [
	(
		"user",
		[
			{"type": "text", "text": prompt_template},
			{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
		],
	)
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | ChatOpenAI(model='gpt-4o-mini') | StrOutputParser()
summarizeImage = chain.batch(images)

if False:
	print(summarizeImage[1])


### Step 3: Load Data and Summary to the VectorDB ###

# Index the Chunks.
vectorStore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

# Storage Layer of Parent Document.
store = InMemoryStore()
id_key = "doc_id"

# Build the Retriever.
retriever = MultiVectorRetriever(
    vectorstore=vectorStore,
    docstore=store,
    id_key=id_key)


# Add the Texts.
txt_id = [str(uuid.uuid4()) for _ in texts]

summary_Text = [
	Document(page_content=summary, metadata={id_key: txt_id[i]}) for i, summary in enumerate(summarizeText) ]
retriever.vectorstore.add_documents(summary_Text)
retriever.docstore.mset(list(zip(txt_id, texts)))


# Add the Tables.
# table_id = [str(uuid.uuid4()) for _ in tables]

# summary_Table = [
# 	Document(page_content=summary, metadata={id_key: table_id[i]}) for i, summary in enumerate(summarizeTable) ]
# retriever.vectorstore.add_documents(summary_Table)
# retriever.docstore.mset(list(zip(table_id, tables)))


# Add the Images.
img_id = [str(uuid.uuid4()) for _ in images]

summary_Image = [
	Document(page_content=summary, metadata={id_key: img_id[i]}) for i, summary in enumerate(summarizeImage) ]
retriever.vectorstore.add_documents(summary_Image)
retriever.docstore.mset(list(zip(img_id, images)))

# Quick Sanity Check. Test Retrieval.

ragDocs = retriever.invoke("Who are the main authors of the Attention paper?")
if False:
	for doc in ragDocs:
	    print(str(doc) + "\n\n")



### Finally, Build the RAG Pipeline ###

start = timeit.default_timer()
ragChain = (
	{
		"context": retriever | RunnableLambda(parseDocs),
		"question": RunnablePassthrough(),
	}
	| RunnableLambda(buildPrompt)
	| ChatOpenAI(model='gpt-4o-mini')
	| StrOutputParser())

ragResponse = ragChain.invoke("What is the Attention mechanism?")
stop = timeit.default_timer()
print("Time taken is: ", round(stop-start, 2), "\n\n", ragResponse)


ragSourceChain = {
		"context": retriever | RunnableLambda(parseDocs),
		"question": RunnablePassthrough(),
	} | RunnablePassthrough().assign(
		response=(
			RunnableLambda(buildPrompt)
			| ChatOpenAI(model="gpt-4o-mini")
			| StrOutputParser() ) )



ragSourceResponse = ragSourceChain.invoke("Describe the Multi-head Attention Architecture?")
print(ragSourceResponse)

if True:
	print("\n\n Context:")
	for text in ragSourceResponse['context']['texts']:
		print(text.text, "\n\n")
	for image in ragSourceResponse['context']['images']:
		displayBase64(image)
