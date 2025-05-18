import chromadb
import torch
import re
import os
import json
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
from transformers import AutoModel, AutoTokenizer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from util.templates import SQ_SYSTEM_MSG, SYSTEM_MSG, generateMessages, chat_prompt
from util.config import COLLECTION_NAME, DB_STORAGE_PATH, EMBEDDING_MODEL, INFERENCE_BASE_URL, LLM, OR_TOKEN
from util.types import ChatRequest

db_client = chromadb.PersistentClient(DB_STORAGE_PATH)
# server_params = StdioServerParameters(
#     command="python",
#     args=[os.path.join(os.path.dirname(__file__), "mcp-server.py")]
# )


# async def get_mcp_tools():
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             tools = await session.list_tools()
#             return [
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": tool.name,
#                         "description": tool.description,
#                         "parameters": tool.inputSchema
#                     }
#                 }
#                 for tool in tools.tools
#             ]


# async def execute_mcp_tool(tool_name: str, args: json):
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             result = await session.call_tool(tool_name, arguments=args)
#             return result


class Retrival:
    __device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __init__(self, client):
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True)
        self.model.to(self.__device)
        self.client = client
        # self.attribute_info = attribute_info

        self.mcp_tools = None
        self.collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                trust_remote_code=True,
                device=self.__device
            )
        )

    async def generateQueryAndFilters(self, question: str):
        # if self.mcp_tools is None:
        #     self.mcp_tools = await get_mcp_tools()
        # print("[MCP TOOLS]", self.mcp_tools)

        messages = generateMessages(SQ_SYSTEM_MSG, question)
        llm_res = self.client.chat.completions.create(
            model=LLM,
            messages=messages,
            temperature=0,
            stream=False,
            # tools=self.mcp_tools
        )
        # if "tool_calls" in llm_res.choices[0].message and llm_res.choices[0].message.tool_calls:
        #     tool_call = llm_res.choices[0].message.tool_calls[0]
        #     print("[Tool call]", tool_call)
        #     result = await execute_mcp_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
        #     translation_res = result.content[0].text
        #     print("[Tool call result]", translation_res)
        #     llm_res = self.getLLMResponse(translation_res, llm_model=llm_model)

        if not llm_res.choices:
            raise Exception(f"LLM Error: {llm_res.error.get('message')}")
        json_str = llm_res.choices[0].message.content
        try:
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                return json.loads(json_string)
            return {'query': question, 'filter': ''}
        except:
            print(f'[Error] LLM response JSON: {json_str}')
            return {'query': question, 'filter': ''}

    async def selfQuery(self, query: str, n_results=5):
        query_json = await self.generateQueryAndFilters(query)
        q_language = query_json.get("language")
        print("[Query JSON]", query_json, "\n")
        if query_json.get("query"):
            q_res = self.query(query_json.get("query"), n_results=n_results)
            return q_res, q_language
        return {'documents': [[]]}, q_language

    def query(self, query: str, n_results: int):
        # TODO: Try to pass query text to collection.query
        query_vector = self.model(**self.tokenizer(text=query, return_tensors="pt").to(
            self.__device)).last_hidden_state[:, 0, :].detach().to(torch.float32).cpu().numpy().flatten()
        return self.collection.query(query_embeddings=query_vector.tolist(), n_results=n_results)

client = OpenAI(base_url=INFERENCE_BASE_URL, api_key=OR_TOKEN)
retrival = Retrival(client)


async def getAnswer(request: ChatRequest):
    last_message = request.messages[-1].content
    print("[Query] : "+last_message)
    sq_res, language = await retrival.selfQuery(last_message, 15)
    print("[SQ Result] :", sq_res)
    return
    context_list = []
    for i, doc in enumerate(sq_res['documents'][0]):
        context_list.append(
            f"{sq_res['metadatas'][0][i]['articleBn'] if language == 'bn' else doc}\n\n\
                ## article={sq_res['metadatas'][0][i]['articleNoBn'] if language == 'bn' else sq_res['metadatas'][0][i]['articleNoEn']}\
                ## topic={sq_res['metadatas'][0][i]['topicBn'] if language == 'bn' else sq_res['metadatas'][0][i]['topicEn']}")
    sq_context_text = "\n\n-----\n\n".join(context_list)
    if sq_context_text:
        print("[Context] :", [sq_res['metadatas'][0][i]['articleNoEn']
              for i in range(len(sq_res['metadatas'][0]))])
    t = chat_prompt.invoke(
        {'question': last_message, 'context': sq_context_text})
    # messages = [
    #     {"role": "system", "content": SYSTEM_MSG},
    #     *[m.model_dump(exclude={'id'}) for m in request.messages[:-1]],
    #     {"role": "user", "content": t.messages[0].content},
    # ]
    messages = generateMessages(
        SYSTEM_MSG,
        t.messages[0].content,
        history=[m.model_dump(exclude={'id'}) for m in request.messages[:-1]]
    )
    stream = client.chat.completions.create(
        model=LLM,
        messages=messages,
        temperature=request.temperature or 0.5,
        stream=request.stream or False
    )
    print("[ANSWER] :", stream.choices[0].message.content)
    return stream.choices[0].message.content
    # for chunk in stream:
    #     print(chunk.choices[0].delta.content, end="")
