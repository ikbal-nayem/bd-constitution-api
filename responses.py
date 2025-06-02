from uuid import uuid4
from openai import OpenAI
from retrival import Retrival
from util.config import INFERENCE_BASE_URL, LLM, OR_TOKEN
from util.generator import generateContextString, generateMessages
from util.types import ChatRequest
from util.templates import SYSTEM_MSG, chat_prompt


client = OpenAI(base_url=INFERENCE_BASE_URL, api_key=OR_TOKEN)
retrival = Retrival(client)


async def getAnswer(request: ChatRequest):
    last_message = request.messages[-1].content
    print("[Query] : "+last_message)
    sq_res, language = await retrival.selfQuery(last_message, 25)
    context_list = []
    for i, doc in enumerate(sq_res['documents'][0]):
        context_str = generateContextString(
            doc, sq_res['metadatas'][0][i], language)
        context_list.append(context_str)
    sq_context_text = "\n\n---\n\n".join(context_list)

    # if sq_context_text:
    #     print("[Context] :", sq_context_text)
    # return sq_context_text
    t = chat_prompt.invoke(
        {'question': last_message, 'contexts': sq_context_text})
    messages = generateMessages(
        SYSTEM_MSG,
        t.messages[0].content,
        history=[m.model_dump(exclude={'id'}) for m in request.messages[:-1]]
    )
    try:
        stream_obj = client.chat.completions.create(
			model=LLM,
			messages=messages,
			temperature=request.temperature or 0.5,
			stream=True
		)
		# message = stream_obj.choices[0].message.content
		# print("[ANSWER] :", message)
		# return {"content": message, "id": uuid4()}
        for chunk in stream_obj:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                finish_reason = chunk.choices[0].finish_reason

                if content:
                    print(content, end='', flush=True)
                    yield content
                if finish_reason == "stop":
                    print("[INFO] LLM stream finished.")
                    break
            else:
                print("[WARN] Received a chunk with no choices.")
    except Exception as e:
        print(f"[ERROR] Error in get_answer_stream: {e}")
        yield f"Error processing your request: {str(e)}\n\n"
