import streamlit as st
import requests
import base64

API_QUERY_URL = "http://127.0.0.1:8000/query"

# CSS Properties
def chat_css():
    st.markdown("""
    <style>
    .chat-container { width: 100%; display: flex; flex-direction: column; margin-top: 10px; }
    .message-row { display: flex; width: 100%; margin: 8px 0; align-items: flex-end; }
    .bot { justify-content: flex-start; }
    .user { justify-content: flex-end; }
    .bubble { max-width: 60%; padding: 12px 16px; border-radius: 18px; font-size: 15px; line-height: 1.5; word-wrap: break-word; }
    .bot .bubble { background-color: #1E1E1E; color: white; border: 2px solid #ED7014; box-shadow: 0 0 6px rgba(237,112,20,0.4); }
    .user .bubble { background-color: #ED7014; color: white; }
    .avatar { font-size: 24px; margin: 0 8px; display: flex; align-items: flex-end; }
    .avatar-img { width: 50px; height: 50px; border-radius: 50%; object-fit: cover; }
    </style>
    """, unsafe_allow_html=True)

# Initialize the session state
def init_chat_state():
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "current_chat" not in st.session_state:
        chat_name = "Chat 1"
        st.session_state.current_chat = chat_name
        st.session_state.conversations[chat_name] = [
            {
                "role": "bot",
                "content": "Welcome to Enterprise Insight Engine! Ask me anything about your uploaded data and get analytics."
            }
        ]

    if "input" not in st.session_state:
        st.session_state.input = ""

# Send message chat UI
def send_message():
    user_text = st.session_state.input.strip()

    if user_text:
        current_chat = st.session_state.current_chat
        st.session_state.conversations[current_chat].append({
            "role": "user",
            "content": user_text
        })
        try:
            resp = requests.post(API_QUERY_URL, json={"question": user_text})
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "success":
                bot_reply = data.get("response")
            else:
                bot_reply = f"Error: {data.get('message')}"

        except Exception as e:
            bot_reply = f"Error: {str(e)}"

        # Save bot reply
        st.session_state.conversations[current_chat].append({
            "role": "bot",
            "content": bot_reply
        })

    st.session_state.input = ""

# Main function to load the chat
def load_chat():

    init_chat_state()
    chat_css()

    # Side Bar
    with st.sidebar:
        st.markdown('<h1 style="color: #ED7014;">💬 Conversations</h1>', unsafe_allow_html=True)

        # New Chat Button
        if st.button(" New Chat"):
            chat_number = len(st.session_state.conversations) + 1
            new_chat_name = f"Chat {chat_number}"

            st.session_state.conversations[new_chat_name] = [
                {
                    "role": "bot",
                    "content": "Welcome to Enterprise Insight Engine! Ask me anything about your uploaded data and get analytics."
                }
            ]

            st.session_state.current_chat = new_chat_name
            st.experimental_rerun()

        st.markdown("---")

        # List Existing Chats
        for chat_name in st.session_state.conversations.keys():
            if st.button(chat_name):
                st.session_state.current_chat = chat_name
                st.experimental_rerun()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    current_chat = st.session_state.current_chat
    messages = st.session_state.conversations[current_chat]

    for msg in messages:
        user_avatar = get_base64_image("images/user.png")
        bot_avatar = get_base64_image("images/chatbot.png")

        avatar = user_avatar if msg["role"] == "user" else bot_avatar
        avatar_html = f'<img src="data:image/png;base64,{avatar}" class="avatar-img">'

        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-row user">
                <div class="bubble">{msg["content"]}</div>
                <div class="avatar">{avatar_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            formatted_text = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="message-row bot">
                <div class="avatar">{avatar_html}</div>
                <div class="bubble">{formatted_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.text_input(
        "",
        key="input",
        placeholder="Type your question and press Enter...",
        on_change=send_message
    )

# Special Function to Load Images
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()