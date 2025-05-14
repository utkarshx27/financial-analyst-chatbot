# 📊 Financial Analyst Chatbot

An interactive chatbot built with **LangGraph**, **LangChain**, **Streamlit**, and the **Financial Modeling Prep API**, designed to help users analyze and understand financial data such as key ratios, cash flow metrics, and income statements for publicly traded companies.

This chatbot leverages the **ReAct (Reasoning + Acting)** technique and **tools-calling functionality** to enhance responses with:
- Real-time financial data retrieval
- Dynamic Python-based calculations
- Web search for financial news or updates
- Logical, explainable reasoning using financial ratios


## 🚀 Features

- 🔍 Fetches balance sheet, income statement, and cash flow data for stock tickers
- 📈 Computes key financial ratios: current ratio, debt/equity, profit margins, EPS, and more
- 🤖 Conversational interface powered by LangGraph + LLMs
- 🛠️ Tool integrations: Python REPL and search
- 📑 Uses markdown formatting for clear explanations

---

## 🧠 Tech Stack

- LangGraph
- LangChain
- Streamlit
- Pandas
- Financial Modeling Prep API
- ChatGroq (Qwen Model)

---

## 📂 Project Structure
```
├── financial_agent.py      # Core logic: data fetching, processing, LangGraph + LLM tools
├── app.py                  # Streamlit frontend for chatbot interface
├── requirements.txt        # All dependencies
├── README.md               # readme file
```
---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/financial-analyst-chatbot.git
cd financial-analyst-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
#### Edit financial_agent.py and replace the API key with your own from Financial Modeling Prep:
```bash
api_key = "YOUR_API_KEY"
```
### 4. Run the app 
```bash
streamlit run app.py
```




