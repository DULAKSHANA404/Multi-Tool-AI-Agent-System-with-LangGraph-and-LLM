from typing import Annotated,TypedDict,Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:float,b:float) -> float:
    """use this function (tool) for add(plus,+) for given two numbers"""
    return a+b

@tool
def sub(a:float,b:float) ->float:
    """use this function (tool) for subtractin(minues,-) for given two numbers"""
    return a-b

@tool
def multi(a:float,b:float):
    """use this function for Multiplication(multi,*,a by b) for given two numbers"""
    return a*b

@tool
def devide(a:float,b:float) -> float:
    """use this functing for devide(/) for given two numbers"""
    return a/b

tools = [sub,add,multi,devide]

llm =ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API")).bind_tools(tools)

def start(state:AgentState)->AgentState:
    prompt = """you are a very helpful ,useful ai assisten name jarvis, u can use any given tool . do not use any not given tools.
            please give the correnct answers for the user. If unsure, ALWAYS use tool. if dont know the answer or not sure please say i dont know."""
    
    responce = llm.invoke([SystemMessage(content=prompt)]+state["messages"])

    if responce.content != "":
        print(f"AI: {responce.content}")

    return {'messages':[responce]}

def should_continue(state:AgentState):
    lst_msg = state["messages"][-1]

    if hasattr(lst_msg,"tool_calls") and lst_msg.tool_calls:
        return "continue"
    
    else:
        return "end"
    

graph = StateGraph(AgentState)
graph.add_node("start",start)
graph.add_node("tools",ToolNode(tools))

graph.add_edge(START,'start')
graph.add_conditional_edges(
    "start",
    should_continue,
    {"continue":"tools","end":END})

graph.add_edge("tools","start")

memory = MemorySaver()
config = {'configurable':{"thread_id":"1"}}
app = graph.compile(checkpointer=memory)

print("---------------started-------------------\n\n")

while True:
    user_input = input('user: ')
    if user_input.lower() in ["quit","exit","q"]:
        break
    results = app.invoke({'messages':HumanMessage(content=user_input)},config=config)


