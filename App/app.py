import streamlit as st
from upload_docs_UI import upload_documents_tab
from chat_UI import load_chat

# Page Configs
st.set_page_config(
    layout="wide",
    page_title="Enterprise Insight Engine",
    page_icon="images/web_icon.png"
)

# CSS properties
st.markdown("""
<style>
/* App main background */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: #ffffff;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1a1a1a;
    color: #ffffff;
}

/* Sidebar text */
.css-1v3fvcr p, .css-1v3fvcr span {
    color: #ffffff;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 24px;
    padding: 14px 28px;
    background-color: #1a1a1a;
    color: #ED7014;
}
button[data-baseweb="tab"]:hover {
    color: #FFA500;
}
button[data-baseweb="tab"]:focus {
    color: #FFA500;
}

/* Buttons */
.stButton>button {
    background-color: #ED7014;
    color: white;
    border: none;
    font-weight: 600;
}

/* Text areas */
.stTextArea>div>div>textarea {
    background-color: #1a1a1a;
    color: #ffffff;
    border: 1px solid #ED7014;
}

/* Dataframe table */
[data-testid="stTable"] {
    color: #ffffff;
}
thead th {
    background-color: #ED7014;
    color: #000000;
}
tbody td {
    background-color: #1a1a1a;
    color: #ffffff;
}

/* Drag & drop file uploader */
[data-testid="stFileUploaderDropzone"] {
    background-color: #0d0d0d;
    border: 2px dashed #ED7014;
    border-radius: 10px;
}

/* Text inside uploader */
[data-testid="stFileUploaderDropzone"] span {
    color: #ffffff;
}

/* Uploaded file pills */
[data-testid="stFileUploaderFile"] {
    background-color: #1a1a1a;
    color: white;
}

/* Tab font size */
button[data-baseweb="tab"] {
    font-size: 24px;
    padding: 14px 28px;
}
            
/* Hide Streamlit menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hide top header bar / padding */
header {visibility: hidden;}
/* Optional: remove extra space it leaves */
body > div:nth-child(1) {padding-top: 0px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""

""", unsafe_allow_html=True)

# Web Page Header
c1, c2 = st.columns([0.25, 2])

with c1:
    st.image("images/web_icon.png", width=110)

with c2:
    st.markdown("""
    <div style="
        font-size: 60px;
        font-weight: 1000;
        color: #ED7014; 
        display: inline-block;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    ">
        Enterprise Insight Engine
    </div>
    """, unsafe_allow_html=True)


# Tabs
UploadDocsTab, QueryTab = st.tabs(["Upload Documents", "Analytical Reasoning"])

# Upload Document Tab
with UploadDocsTab:
    upload_documents_tab()

# Query tab for chatting
with QueryTab:
    load_chat()