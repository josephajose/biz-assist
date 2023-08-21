import streamlit as st
from dotenv import load_dotenv
from io import StringIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_text(uploaded_files):
    text_list = []
    for file in uploaded_files:
        stringio = StringIO(file.getvalue().decode("utf-8"))
        text_list.append(stringio.read())
    return ''.join(text_list)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory 
    )

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="Biz Assist")
    st.write(css, unsafe_allow_html=True)

    st.header("Biz Assist")
    st.subheader("Your custom personal business assistant!")
    user_question = st.text_input("What can I help you with today?")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Data")
        
        uploaded_file = st.file_uploader(
            "What files would you like me to take into consideration?",
            accept_multiple_files=True)
        
        if st.button("Process Files"):
            with st.spinner("Processing"):
                text_blob = get_text(uploaded_file)
                text_chunks = get_text_chunks(text_blob)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)
            st.success('Done!')

if __name__ == '__main__':
    main()