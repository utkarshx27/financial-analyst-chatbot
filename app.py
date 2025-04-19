import streamlit as st
from financial_agent import get_response
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

st.set_page_config(page_title="Financial Analyst Chatbot", layout="wide")
st.title("📊 Financial Analyst Chatbot")
st.markdown("Ask about financial ratios, earnings trends, or stock tickers!")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful and expert financial analyst.You can help user by providing financial news, financial data analysis, etc. Always explain your reasoning using financial ratios. Prefer markdown formatting.")
    ]



user_input = st.chat_input("Type your financial question (e.g., 'Analyze AAPL income statement')")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    messages = get_response(st.session_state.chat_history)
    st.session_state.chat_history = messages 

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
