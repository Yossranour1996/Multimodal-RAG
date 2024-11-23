import os
import base64
import io
import re
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def categorize_elements(raw_pdf_elements):
    text_elements = []
    table_elements = []
    for element in raw_pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text_elements.append(str(element))
        elif 'Table' in str(type(element)):
            table_elements.append(str(element))
    return text_elements, table_elements


def generate_text_summaries(texts, tables, summarize_texts=False):
    prompt_text = """Summarize this element (table or text): {element}"""
    prompt = PromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=1024)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1}) if summarize_texts else texts
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1}) if tables else []

    return text_summaries, table_summaries


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    model_vision = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=1024)
    msg = model_vision.invoke([HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}])])
    return msg.content


def generate_img_summaries(path):
    img_base64_list, image_summaries = [], []
    prompt = "Summarize this image for retrieval."
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    return img_base64_list, image_summaries


def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries: add_documents(retriever, text_summaries, texts)
    if table_summaries: add_documents(retriever, table_summaries, tables)
    if image_summaries: add_documents(retriever, image_summaries, images)

    return retriever


def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    image_signatures = {b"\xFF\xD8\xFF": "jpg", b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png", b"\x47\x49\x46\x38": "gif", b"\x52\x49\x46\x46": "webp"}
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    b64_images, texts = [], []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
