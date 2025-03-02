import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch

model_id = 'model/Mistral-7B-v0.1'
device = 'cpu'
torch.device('cpu')

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map = device
    )
    torch.device('cpu')
    generation_pipeline = pipeline(
        task="text-generation",
        model = model,
        tokenizer = tokenizer
    )
    return generation_pipeline

generation_pipeline = load_model()

st.set_page_config(
    page_title="Law Assist",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
    <style>
        body{
            background-color:black;
            color:white
        }
        .main{
            background-color: black;
        }
        h1{
            color:red;
            text-align:center;
        }
        textarea, input{
            background-color: #222;
            color:white;
        }
        .block-container{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Marketplace"])

if "conversations" not in st.session_state:
    st.session_state["conversations"] = []

if page == "Chatbot":
    st.markdown("<h1>AI Assist</h1>", unsafe_allow_html=True)

    if st.session_state['conversations']:
        st.subheader("Chat History: ")
        for i, chat in enumerate(st.session_state["conversations"], 1):
            st.markdown("----")
    else:
        st.info("No chats yet. Start the conversation below!")

    # st.write("/n" * 15)
    query = st.text_input("Type your query here....")
    if query:
        with st.spinner("Generating response...."):
            output = generation_pipeline(query, max_new_tokens=100)
            ai_response = output[0]['generated_text']

        st.session_state["conversations"].append({
            "query": query,
            "response": ai_response
        })
        st.success(f"Your query has been received: {query}")
        st.markdown()

    elif page=="Marketplace":
        st.markdown("<h1>Marketplace</h1>", unsafe_allow_html=True)
        st.info("Marketplace feature coming soon")