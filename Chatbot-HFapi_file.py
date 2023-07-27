#pip install openai langchain docx2txt pypdf streamlit tiktoken 
#pip install huggingface python-dotenv sentence_transformers faiss-cpu streamlit-chat

import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import random
import time

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    #embeddings = OpenAIEmbeddings()
    #vector_store = Chroma.from_documents(chunks, embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain import HuggingFaceHub
    
    repo_id = "tiiuae/falcon-7b-instruct" 
    llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)
    #llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']



if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    #st.image('img.png')
    st.subheader('LLM Question-Answering Application')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        huggingface_api_key = st.text_input('Huggingface API Key:', type='password')
        if huggingface_api_key:
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                #tokens, embedding_cost = calculate_embedding_cost(chunks)
                #st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')
        clear_button = st.sidebar.button("Clear Conversation", key="clear")


    if clear_button:        
        st.session_state['messages'] = []
        st.session_state['vs'] = []     


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    print(st.session_state.messages)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if q := st.chat_input("Ask a question about the content of your file:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": q})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(q)
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                assistant_response = ask_and_get_answer(vector_store, q, k)
                
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


#     # user's question text input widget
#     q = st.text_input('Ask a question about the content of your file:')
#     if q: # if the user entered a question and hit enter
#         if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
#             vector_store = st.session_state.vs
#             #st.write(f'k: {k}')
#             answer = ask_and_get_answer(vector_store, q, k)

#             # text area widget for the LLM answer
#             st.text_area('LLM Answer: ', value=answer)

#             st.divider()

#             # if there's no chat history in the session state, create it
#             if 'history' not in st.session_state:
#                 st.session_state.history = ''

#             # the current question and answer
#             value = f'Q: {q} \nA: {answer}'

#             st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
#             h = st.session_state.history

#             # text area widget for the chat history
#             #st.text_area(label='Chat History', value=h, key='history', height=400)

# # run the app: streamlit run ./chat_with_documents.py

