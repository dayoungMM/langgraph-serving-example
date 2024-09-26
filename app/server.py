#!/usr/bin/env python

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from flows.graph import llm_client, graph
from flows.implements import MessageState
from app.type import ReqGraphExecution, ResGraphExecution

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

joke_prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

joke_chain = joke_prompt | llm_client

add_routes(
    app,
    joke_chain,
    path="/joke",
)


def create_state(input_data: dict) -> MessageState:
    return MessageState(
        messages=[HumanMessage(content=input_data.get("query"))],
        prev_node="user",
    )


def parse_result(state: dict) -> ResGraphExecution:
    return ResGraphExecution(
        content=state["messages"][-1].content,
        message_history=state["messages"],
    )


graph_execution_chain = create_state | graph | parse_result


add_routes(
    app,
    graph_execution_chain,
    path="/graph",
    input_type=ReqGraphExecution,
    output_type=ResGraphExecution,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)
