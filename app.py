import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="YouTube Groq Chatbot")
st.title("🎥 YouTube Chatbot (Powered by Groq)")

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        return None


youtube_url = st.text_input("Paste YouTube Video URL")


if youtube_url:

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi().list(video_id)

        transcript = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )

        transcript_data = transcript.fetch()

        text = " ".join([item.text for item in transcript_data])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.create_documents([text])

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(docs, embeddings)

        llm = ChatGroq(
            model="llama-3.1-8b-instant"
        )

        st.success("✅ Video processed successfully!")
        st.write("Ask your question below 👇")

        query = st.text_input("Ask your question")

        if query:
            retriever = vector_store.as_retriever()
            retrieved_docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            prompt = ChatPromptTemplate.from_template(
                """
                Answer the question based only on the context below.

                Context:
                {context}

                Question:
                {question}
                """
            )

            chain = prompt | llm

            result = chain.invoke({
                "context": context,
                "question": query
            })

            st.write(result.content)

    except Exception as e:
        st.error(f"Error: {str(e)}")
