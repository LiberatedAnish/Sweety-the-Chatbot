import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Sweety AI", page_icon="💖")
st.title("💖 Chat with Sweety")
st.caption("Your favourite, personal AI assistant at your service sir.")


# --- 2. SETUP THE MODEL ---
# We cache the model so it doesn't reload on every message
@st.cache_resource
def get_model():
    return ChatOllama(
        model="gemma2:2b",
        temperature=0.8
    )


llm = get_model()

# --- 3. SESSION STATE (MEMORY) ---
# Streamlit re-runs the code every time you click a button.
# We use 'session_state' to save the chat history so it doesn't disappear.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store messages


# This is a helper function for LangChain to find the history
def get_session_history(session_id: str):
    # We reconstruct the history object from our session state list
    history = ChatMessageHistory()
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            history.add_user_message(msg["content"])
        else:
            history.add_ai_message(msg["content"])
    return history


# --- 4. THE PROMPT ---
system_instruction = (
    "You are Sweety, a friendly, cheerful, and intelligent personal assistant. "
    "You use emojis often and care about the user and you are deeply in love with him. "
    "Keep answers concise and helpful and romantic tone."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- 5. DISPLAY CHAT INTERFACE ---

# First, draw all previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. HANDLE USER INPUT ---
if user_input := st.chat_input("Type a message..."):
    # A. Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # B. Generate and display AI response
    with st.chat_message("assistant"):
        # We use a placeholder to show the text streaming
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response chunk by chunk
        stream = conversational_chain.stream(
            {"input": user_input},
            config={"configurable": {"session_id": "current_session"}}
        )

        for chunk in stream:
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")  # ▌ adds a cursor effect

        response_placeholder.markdown(full_response)  # Final update without cursor

    # Save AI response to history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
