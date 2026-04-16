import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.gemini_client import GeminiClient


TEST_CONTEXT = """[Source: Week 10/11 Milestones]
Week 10/11 Milestones
# Major Milestones
- RAG Pipeline Accuracy Improvements
- User Data Privacy & Access Control
- Self-Updating Knowledge Base
- Connect WorkMate account to Notion Workspace

# Task List
The following tasks are for the remaining 5 weeks before the Expo on April 28th.
Task | Who | Priority (1-5) | Ready for CP#3?
RAG Pipeline Accuracy Improvements | @Anonymous | Priority 4 | Probably
User Data Privacy & Access Control | @Anonymous | Priority 4 | No

[Source: Findings & Updates]
Findings & Updates
BM25 hybrid search with RRF (Reciprocal Rank Fusion) implemented.
Retrieval Improvements: removed MarkdownHeaderTextSplitter.
re retrieved content can reach Gemini without truncation — BM25 vs Sparse Embedding.
"""


def test_sync(client: GeminiClient) -> str:
    print("\n=== Test 1: Sync response ===")
    question = "What are the major milestones for week 10/11?"

    response = client.ask_workmate(TEST_CONTEXT, question)

    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 50, "Response too short — likely an error"
    assert "Source" in response, "Response must contain a citation"
    assert "Confidence" in response, "Response must contain a confidence level"

    print(f"Question : {question}")
    print(f"Response :\n{response}")
    print("sync ✓")
    return response


async def test_stream(client: GeminiClient) -> None:
    print("\n=== Test 2: Streaming response ===")
    question = "What retrieval improvements were made?"

    print(f"Question : {question}")
    print("Streaming tokens: ", end="", flush=True)

    full_response = ""
    token_count = 0

    async for token in client.ask_workmate_stream(TEST_CONTEXT, question):
        print(token, end="", flush=True)
        full_response += token
        token_count += 1

    print(f"\n\nTokens received : {token_count}")
    assert token_count > 1, "Expected multiple tokens — streaming not working"
    assert len(full_response) > 50, "Streamed response too short"
    assert "Source" in full_response, "Streamed response must contain a citation"
    print("stream ✓")


def test_fallback(client: GeminiClient) -> None:
    print("\n=== Test 3: Fallback phrase ===")
    question = "What is the weather in Paris today?"

    response = client.ask_workmate(TEST_CONTEXT, question)
    print(f"Question : {question}")
    print(f"Response : {response}")

    assert "cannot find" in response.lower(), \
        f"Expected fallback phrase, got: {response[:100]}"
    print("fallback ✓")


def test_grounding(client: GeminiClient) -> None:
    print("\n=== Test 4: Grounding — model must not answer from training data ===")
    empty_context = "[Source: Empty]\nNo relevant content found."
    question = "What is the capital of France?"

    response = client.ask_workmate(empty_context, question)
    print(f"Question : {question}")
    print(f"Response : {response}")

    assert "cannot find" in response.lower() or "Paris" not in response, \
        "Model answered from training data — grounding is not working"
    print("grounding ✓")


def main():
    client = GeminiClient()

    test_sync(client)
    asyncio.run(test_stream(client))
    test_fallback(client)
    test_grounding(client)

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    main()