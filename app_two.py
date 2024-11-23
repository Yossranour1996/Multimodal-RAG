from unstructured.partition.pdf import partition_pdf  # required popplers and tesseract
import os
import base64
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils_two import *
import io
import pytesseract
import streamlit as st
from dotenv import load_dotenv

# Load API keys and environment variables
load_dotenv()
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def plt_img_base64(b64data):
    """
    Display base64-encoded image data
    """
    try:
        decoded_bytes = base64.b64decode(b64data)
        image = Image.open(io.BytesIO(decoded_bytes))
        st.image(image, caption='Relevant Image', use_column_width=True)
    except Exception as e:
        st.write(f"Error displaying image: {e}")


def extract_pdf_path(uploaded_file):
    """
    Extracts the file path from the UploadedFile object.
    """
    if hasattr(uploaded_file, 'name'):
        return uploaded_file.name
    else:
        st.warning("Unable to determine file path. Using a default filename.")
        return "default.pdf"


def img_prompt_func(data_dict):
    """
    Joins the context into a single string for the model prompt
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = [{"type": "text", "text": f"User-provided question: {data_dict['question']}\n\nText and/or tables:\n{formatted_texts}"}]

    # Add images if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """
    model_vision = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model_vision
        | StrOutputParser()
    )

    return chain


@st.cache_data
def data_loader(pdf_path):
    # Extract images, text, and tables from the PDF
    image_path = "./figures"
    pdf_elements = partition_pdf(pdf_path, chunking_strategy="by_title", extract_images_in_pdf=True, infer_table_structure=True, max_characters=3000, new_after_n_chars=2800, combine_text_under_n_chars=2000, image_output_dir_path=image_path)

    # Separate texts and tables
    texts, tables = categorize_elements(pdf_elements)
    # Get text & table summaries
    text_summaries, table_summaries = generate_text_summaries(texts[:19], tables, summarize_texts=True)
    # Generate image summaries
    img_base64_list, image_summaries = generate_img_summaries(image_path)

    return text_summaries, texts, table_summaries, tables, image_summaries, img_base64_list


@st.cache_resource
def retriever_func(text_summaries, texts, table_summaries, tables, image_summaries, img_base64_list):
    vectorstore = Chroma(
        collection_name="mm_rag_gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="./chroma_db"
    )

    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    stored_documents = vectorstore.get()
    st.write("Documents in Chroma vectorstore:", stored_documents)

    return retriever_multi_vector_img


def main():
    st.title("Multi-Modal RAG ResearcherQA Bot")

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if pdf_file is not None:
        pdf_path = extract_pdf_path(pdf_file)
        if 'retriever_multi_vector_img' not in st.session_state:
            pdf_data = data_loader(pdf_path)
            retriever_multi_vector_img = retriever_func(*pdf_data)
            st.session_state.retriever_multi_vector_img = retriever_multi_vector_img

    user_input = st.text_input("Enter your question:")
    generate_button = st.button("Generate Answer")

    if generate_button and pdf_file is not None:
        retriever_multi_vector_img = st.session_state.retriever_multi_vector_img
        query = f"{user_input}"
        docs = retriever_multi_vector_img.invoke(query, config={"limit": 1})

        with st.expander("Relevant Content"):
            for doc in docs:
                if is_image_data(doc):
                    plt_img_base64(doc)
                else:
                    st.write(doc)
                
        chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
        st.text_area("Final result from LLM", chain_multimodal_rag.invoke(query))


if __name__ == "__main__":
    main()
