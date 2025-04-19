from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
import pandas as pd
import requests
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# fitching financial data
# https://site.financialmodelingprep.com/developer/docs#income-statements-financial-statements


def calculate_financial_ratios(df):
    dfc = df.copy()

    dfc['current_ratio'] = dfc['totalcurrentassets'] / dfc['totalcurrentliabilities']
    dfc['quick_ratio'] = (dfc['cashandcashequivalents'] + dfc['shortterminvestments'] + dfc['netreceivables']) / dfc['totalcurrentliabilities']

    dfc['debt_to_equity_ratio'] = dfc['totalliabilities'] / dfc['totalstockholdersequity']
    dfc['debt_ratio'] = dfc['totalliabilities'] / dfc['totalassets']
    dfc['net_debt_to_equity_ratio'] = (dfc['totaldebt'] - dfc['cashandcashequivalents']) / dfc['totalstockholdersequity']

    dfc['equity_ratio'] = dfc['totalstockholdersequity'] / dfc['totalassets']
    return dfc


def calculate_income_statement_ratios(df):
    dfc = df.copy()

    dfc['gross_profit_margin'] = dfc['grossprofit'] / dfc['revenue']
    dfc['operating_profit_margin'] = dfc['operatingincome'] / dfc['revenue']
    dfc['net_profit_margin'] = dfc['netincome'] / dfc['revenue']
    dfc['ebitda_margin'] = dfc['ebitda'] / dfc['revenue']

    dfc['eps'] = dfc['netincome'] / dfc['weightedaverageshsout']
    dfc['eps_diluted'] = dfc['netincome'] / dfc['weightedaverageshsoutdil']

    return dfc


def calculate_cash_flow_metrics(cash_flow_df, income_df=None, balance_df=None):
    df = cash_flow_df.copy()

    required_columns = ['netcashprovidedbyoperatingactivities', 'freecashflow',
                        'dividendspaid', 'netincome', 'investmentsinpropertyplantandequipment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing necessary columns in cash flow data")

    if balance_df is not None:
        total_liabilities = balance_df['totalliabilities'].iloc[0]
        df['operating_cash_flow_ratio'] = df['netcashprovidedbyoperatingactivities'] / total_liabilities

    if income_df is not None:
        revenue = income_df['revenue'].iloc[0]
        df['cash_flow_margin'] = df['netcashprovidedbyoperatingactivities'] / revenue

    df['reinvestment_ratio'] = df['investmentsinpropertyplantandequipment'] / df['netcashprovidedbyoperatingactivities']

    df['dividend_payout_ratio'] = df['dividendspaid'] / df['freecashflow']

    if income_df is not None:
        df['fcf_to_revenue'] = df['freecashflow'] / revenue

    df['cash_conversion_efficiency'] = df['netcashprovidedbyoperatingactivities'] / df['netincome']

    return df
def get_financial_data(ticker, data_type):
    base_url = 'https://financialmodelingprep.com/api'
    api_key = "Y4CUC0IYYYX093DPnJeSqarUMJEoER6S"
    url = f'{base_url}/v3/{data_type}/{ticker}?period=annual&apikey={api_key}'

    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
        data = response.json()
        if not data or 'error' in data:
            raise ValueError(f"No data found or API error: {data.get('error', 'Unknown error')}")

        df = pd.DataFrame(data)
        df = std_financial_data(df)

        if data_type == 'balance-sheet-statement':
            df = calculate_financial_ratios(df)
        elif data_type == 'income-statement':
            df = calculate_income_statement_ratios(df)
        elif data_type == 'cash-flow-statement':

            income_df = get_financial_data(ticker, 'income-statement')
            balance_df = get_financial_data(ticker, 'balance-sheet-statement')
            df = calculate_cash_flow_metrics(df, income_df=income_df, balance_df=balance_df)

        df.to_csv(f'{ticker}_{data_type}_final.csv', index=False)
        return df.to_markdown(index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def std_financial_data(df):
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")
    dfc = df.copy()

    if 'calendarYear' not in dfc.columns:
        raise KeyError("'calendarYear' column is missing.")

    dfc['calendarYear'] = pd.to_numeric(dfc['calendarYear'], errors='coerce')
    if dfc['calendarYear'].isna().any():
        raise ValueError("Invalid 'calendarYear' entries.")

    current_year = dfc['calendarYear'].max()
    dfc['calendarYear'] = dfc['calendarYear'].apply(
        lambda x: f"t-{current_year - x}" if x < current_year else "t"
    )

    to_remove = ['cik', 'symbol', 'fillingDate', 'acceptedDate', 'calendarYear', 'link', 'finalLink']
    dfc = dfc.drop(columns=[col for col in to_remove if col in dfc.columns], errors='ignore')

    dfc.columns = dfc.columns.str.lower().str.replace(' ', '_')
    return dfc



python_repl = PythonREPL()
tavily = TavilySearchResults()

class FinancialDataInput(BaseModel):
    ticker: str
    data_type: str

get_financial_data_tool = StructuredTool.from_function(
    name="get_financial_data",
    description="Fetches financial data for a given stock ticker symbol. Valid data_type values are: 'balance-sheet-statement', 'income-statement', 'cash-flow-statement'.",
    func=get_financial_data,
    args_schema=FinancialDataInput,
)

tools = [
    get_financial_data_tool,
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute Python code, such as calculations or DataFrame operations.",
        func=python_repl.run
    ),
    tavily
]

llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def tools_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tools_calling_llm", tools_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tools_calling_llm")
builder.add_conditional_edges("tools_calling_llm", tools_condition)
builder.add_edge("tools", "tools_calling_llm")

graph = builder.compile()


def get_response(chat_history: list):
    """Invokes LangGraph with chat history."""
    result = graph.invoke({"messages": chat_history})
    return result["messages"]
