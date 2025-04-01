from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("Translator")

@mcp.tool()

async def translate(text: str, source_language: str, target_language: str) -> str:
  """Translate text from source language to target language."""
  async with httpx.AsyncClient() as client:
    response = await client.get(f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_language}&tl={target_language}&dt=t&q={text}")
  return "Translated text => "+response.json()[0][0][0]


if __name__ == "__main__":
  print("MCP server is running...")
  mcp.run(transport="stdio")