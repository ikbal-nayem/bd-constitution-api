from gc import collect
import chromadb
import torch
import re
import json
import asyncio
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from util.generator import generateMessages
from util.templates import SQ_SYSTEM_MSG
from util.config import COLLECTION_NAME, DB_STORAGE_PATH, EMBEDDING_MODEL, LAND_COLLECTION_NAME, LLM

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
    __device = str(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))

    def __init__(self, client):
        self.client = client
        self.mcp_tools = None
        self.collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.__embeddingFunction()
        )
        self.land_collection = db_client.get_or_create_collection(
            name=LAND_COLLECTION_NAME,
            embedding_function=self.__embeddingFunction()
        )

    def __embeddingFunction(self):
        return SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
            device=self.__device
        )

    async def generateQueryAndFilters(self, question: str):
        # if self.mcp_tools is None:
        #     self.mcp_tools = await get_mcp_tools()
        # print("[MCP TOOLS]", self.mcp_tools)

        messages = generateMessages(SQ_SYSTEM_MSG, question)
        llm_res = await asyncio.to_thread(self.client.chat.completions.create,
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
            print(llm_res)
            raise Exception(f"LLM Error: {llm_res.error.get('message')}")
        json_str = llm_res.choices[0].message.content
        try:
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                return json.loads(json_string)
            return {'query': question, 'language': 'en'}
        except:
            print(f'[Error] LLM response JSON: {json_str}')
            return {'query': question, 'language': 'en'}

    async def selfQuery(self, query: str, act:str, n_results=5):
        query_json = await self.generateQueryAndFilters(query)
        q_language = query_json.get("language")
        print("[Query JSON]", query_json, "\n")
        if query_json.get("query") or query_json.get("sections"):
            q_res = self.query(query_json.get("query"), act, query_json.get(
                "sections"), n_results=n_results)
            return q_res, q_language
        return {'documents': [[]]}, q_language

    def query(self, query: list[str], sections: dict, act:str, n_results: int):
        collection = self.land_collection if act == 'LAND' else self.collection
        if len(sections):
            where = {"section_no_en": {"$in": sections}}
            return collection.query(query_texts=query, where=where, n_results=n_results)
        else:
            return collection.query(query_texts=query, n_results=n_results)
