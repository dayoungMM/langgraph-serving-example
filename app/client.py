import os
from langserve import RemoteRunnable
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import ssl
import httpx

load_dotenv(".env")

client = None
request_with_sk_ssl = os.environ.get("REQUEST_WITH_SK_SSL", False)
if request_with_sk_ssl:
    cert_file = os.environ.get("SSL_CERT_FILE", "./ssl_cacert.pem")
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=cert_file)
    client = httpx.Client(verify=ssl_context)

llm_client = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    timeout=60,
    client=client,
)

if __name__ == "__main__":
    ####### 단순 실행 ######
    connection_test = llm_client.invoke("hello")

    joke_chain = RemoteRunnable("http://localhost:8080/joke/")

    parrots_joke = joke_chain.invoke({"topic": "parrots"})
    print(">>> Parrots Joke")
    print(parrots_joke)
    print("---")

    agent = RemoteRunnable("http://localhost:8080/graph/")

    result = agent.invoke({"query": "What is the capital of France?"})
    print(">>> Capital of France")
    print(result)
    print("---")
    ####### Chaining 해서 실행  ######
    prompt_template = """
    Create a quiz show question. With 4 multiple choices.
    Format must be like this:
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "..."

    Let's start! Question is {content}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = agent | prompt | llm_client

    final_result = chain.invoke({"query": "What is the capital of France?"})
    print(">>> Capital of Korea")
    print(final_result)
    print("---")
