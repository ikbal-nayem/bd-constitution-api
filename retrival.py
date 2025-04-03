import chromadb
import torch
import re
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from huggingface_hub import InferenceClient

from util.templates import SQ_SYSTEM_MSG, SYSTEM_MSG, self_query_prompt, metadata_field_info, chat_prompt
from util.config import DB_STORAGE_PATH, EMBEDDING_MODEL, LLM
from util.types import ChatRequest

load_dotenv()

db_client = chromadb.PersistentClient(DB_STORAGE_PATH)
server_params = StdioServerParameters(
    command="python",
    args=[os.path.join(os.path.dirname(__file__), "mcp-server.py")]
)


async def get_mcp_tools():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema  # Ensure this follows JSON schema
                    }
                }
                for tool in tools.tools
            ]


class Retrival:
    __device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, EMBEDDING_MODEL, client, collection_name: str, attribute_info: dict = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, trust_remote_code=True)
        self.model.to(self.__device)
        self.client = client
        self.attribute_info = attribute_info

        self.mcp_tools = None
        self.collection = db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                trust_remote_code=True,
                device=self.__device
            )
        )

    def generateQueryMsg(self, temp):
        return [
            {"role": "system", "content": SQ_SYSTEM_MSG},
            {"role": "user", "content": temp.messages[0].content},
        ]

    async def generateQueryAndFilters(self, question: str, attribute_info: dict = None, llm_model: str = LLM):
        pipe = (self_query_prompt | self.generateQueryMsg)
        messages = pipe.invoke({'question': question, 'attribute_info': json.dumps(
            attribute_info or self.attribute_info)})

        if self.mcp_tools is None:
            self.mcp_tools = await get_mcp_tools()
            print("MCP tools ==> ", self.mcp_tools)

        llm_res = self.client.chat_completion(
            model=llm_model,
            messages=messages,
            temperature=0.5,
            stream=False,
            tools=self.mcp_tools
        )
        print("LLM response ==> ", llm_res)
        json_str = llm_res.choices[0].message.content
        try:
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                return json.loads(json_string)
            return {'query': question, 'filter': ''}
        except:
            print('LLM response JSON parse not error: '+json_str)
            return {'query': question, 'filter': ''}

    async def selfQuery(self, query: str, n_results=5):
        query_json = await self.generateQueryAndFilters(query)
        q_language = query_json.get("language")
        print("Query JSON ==> ", query_json, "\n")
        if query_json.get("query"):
            q_res = self.query(query_json.get("query"), n_results=n_results)
            return q_res, q_language
        return {'documents': [[]]}, q_language

    def query(self, query: str, n_results: int):
        query_vector = self.model(**self.tokenizer(text=query, return_tensors="pt").to(
            self.__device)).last_hidden_state[:, 0, :].detach().to(torch.float32).cpu().numpy().flatten()
        return self.collection.query(query_embeddings=query_vector.tolist(), n_results=n_results)


client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
retrival = Retrival(EMBEDDING_MODEL, client,
                    collection_name="bd-constitution", attribute_info=metadata_field_info)


async def getAnswer(request: ChatRequest):
    last_message = request.messages[-1].content
    print("[Query] : "+last_message)
    sq_res, language = await retrival.selfQuery(last_message, 10)
    context_list = []
    for i, doc in enumerate(sq_res['documents'][0]):
        context_list.append(
            f"{sq_res['metadatas'][0][i]['articleBn'] if language == 'bn' else doc}\n\n\
                ## article={sq_res['metadatas'][0][i]['articleNoBn'] if language == 'bn' else sq_res['metadatas'][0][i]['articleNoEn']}\
                ## topic={sq_res['metadatas'][0][i]['topicBn'] if language == 'bn' else sq_res['metadatas'][0][i]['topicEn']}")
    sq_context_text = "\n\n-----\n\n".join(context_list)
    t = chat_prompt.invoke(
        {'question': last_message, 'context': sq_context_text})
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        *[m.model_dump(exclude={'id'}) for m in request.messages[:-1]],
        {"role": "user", "content": t.messages[0].content},
    ]
    stream = client.chat.completions.create(
        model=LLM,
        messages=messages,
        temperature=request.temperature or 0.5,
        stream=request.stream or False
    )
    # print("Answer ==> ", stream.choices[0].message.content)
    return stream.choices[0].message.content
    # for chunk in stream:
    #     print(chunk.choices[0].delta.content, end="")
