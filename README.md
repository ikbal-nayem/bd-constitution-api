---
license: apache-2.0
title: BD Constitution AI
sdk: docker
emoji: 👀
colorFrom: yellow
colorTo: green
short_description: Bangladesh constitution law Agentic RAG implementation API
---

*Chat URL*: `/chat`


*Method*: `POST`


*Content-Type*: `application/json`


*Request Body*:
```json
{
  "messages": {
    "role": "user" | "assistant",
    "content": "string",
    "id?": "string"
  }[],
  "max_tokens?": "number",
  "temperature?": "number",
  "stream?": "boolean"
}
```
