import streamlit as st
import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# Function to convert session state messages to the expected format
def convert_messages_to_expected_format(messages):
    return [
        HumanMessage(content=message["content"], type="human")
        for message in messages if message["role"] == "user"
    ]

openai.api_key = st.secrets["openai_key"]


st.title("Chatte per KI mit dem EU AI ACT")


if "data" not in st.session_state:
    st.session_state["data"] = PyPDFLoader("aiact.pdf").load()

if "all_splits" not in st.session_state:
    st.session_state["all_splits"] = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(st.session_state.data)

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = FAISS.from_documents(documents=st.session_state.all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai.api_key))
retriever = st.session_state.vectorstore.as_retriever(k=4)

# Chat model setup
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat_model, question_answering_prompt)
retrieval_chain = RunnablePassthrough.assign(
    context=lambda params: retriever.invoke(params["messages"][-1].content)
).assign(
    answer=document_chain,
)

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and process
if user_input := st.chat_input("Deine Frage:"):
    # Add user message to session state history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    # Convert session state messages to the format expected by ChatMessageHistory
    converted_messages = convert_messages_to_expected_format(st.session_state.messages)
    chat_history = ChatMessageHistory(messages=converted_messages)
    
    # Invoke the document retrieval and response generation chain
    response = retrieval_chain.invoke({"messages": chat_history.messages})
    
    # Add AI response to session state history
    st.session_state.messages.append({"role": "ai", "content": response["answer"]})
    
    # Update the display with AI response
    with st.chat_message("ai"):
        st.markdown(response["answer"])
