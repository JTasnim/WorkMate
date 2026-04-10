import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.notion.client import NotionClient
from src.backend.notion.block_parser import BlockParser
from src.backend.notion.models.document import NotionDocument


def main():
    client = NotionClient()
    parser = BlockParser()

    # ─────────────────────────────────────────
    # Test 1 — API connection + workspace search
    # ─────────────────────────────────────────
    print("=== Test 1: API connection ===")
    pages = client.search(filter_type="page")
    assert len(pages) > 0, "No pages found — check your integration token and page sharing"
    print(f"Notion API connected ✓")
    print(f"Workspace pages found: {len(pages)}")

    # ─────────────────────────────────────────
    # Test 2 — fetch a single page
    # ─────────────────────────────────────────
    print("\n=== Test 2: Fetch single page ===")
    sample_page = pages[0]
    page_id = sample_page["id"]
    page_data = client.get_page(page_id)

    assert "id" in page_data, "Page must have an id"
    assert "properties" in page_data, "Page must have properties"
    print(f"Page ID     : {page_data['id']}")
    print(f"Object type : {page_data['object']}")

    # Extract title — it lives inside properties under a 'title' type property
    title = ""
    for prop_name, prop_value in page_data["properties"].items():
        if prop_value.get("type") == "title":
            rich_text = prop_value["title"]
            title = "".join(rt["plain_text"] for rt in rich_text)
            break
    print(f"Title       : '{title}'")

    # ─────────────────────────────────────────
    # Test 3 — fetch and parse blocks
    # ─────────────────────────────────────────
    print("\n=== Test 3: Fetch and parse blocks ===")
    blocks = client.get_block_children(page_id)
    print(f"Raw blocks fetched: {len(blocks)}")

    parsed_lines = []
    for block in blocks:
        text = parser.parse_block(block)
        if text.strip():
            parsed_lines.append(text)

    print(f"Non-empty parsed lines: {len(parsed_lines)}")
    print("First 3 parsed lines:")
    for line in parsed_lines[:3]:
        print(f"  '{line[:80]}'")

    assert len(parsed_lines) > 0, "No text parsed — check block_parser.py"

    # ─────────────────────────────────────────
    # Test 4 — build a NotionDocument
    # ─────────────────────────────────────────
    print("\n=== Test 4: Build NotionDocument ===")
    content = "\n".join(parsed_lines)

    doc = NotionDocument(
        id=page_id,
        title=title,
        content=content,
        parent_id=page_data.get("parent", {}).get("page_id", "workspace"),
        url=page_data.get("url", ""),
        source_type=page_data.get("object", "page"),
        properties=page_data.get("properties", {}),
        created_time=page_data.get("created_time", ""),
        last_edited_time=page_data.get("last_edited_time", ""),
    )

    doc_dict = doc.to_dict()
    assert "id" in doc_dict
    assert "title" in doc_dict
    assert "content" in doc_dict
    assert "parent_id" in doc_dict
    assert "url" in doc_dict

    print(f"NotionDocument created ✓")
    print(f"  id       : {doc.id}")
    print(f"  title    : '{doc.title}'")
    print(f"  content  : {len(doc.content)} chars")
    print(f"  parent_id: {doc.parent_id}")
    print(f"  url      : {doc.url}")

    print(f"  source_type      : {doc.source_type}")
    print(f"  properties      : {doc.properties}")
    print(f"  created_time      : {doc.created_time}")
    print(f"  last_edited_time      : {doc.last_edited_time}")

    # ─────────────────────────────────────────
    # Test 5 — block parser types
    # ─────────────────────────────────────────
    print("\n=== Test 5: Block parser type coverage ===")
    mock_blocks = [
        {"type": "paragraph",           "paragraph":           {"rich_text": [{"plain_text": "Hello world"}]}},
        {"type": "heading_1",           "heading_1":           {"rich_text": [{"plain_text": "Title"}]}},
        {"type": "heading_2",           "heading_2":           {"rich_text": [{"plain_text": "Subtitle"}]}},
        {"type": "heading_3",           "heading_3":           {"rich_text": [{"plain_text": "Section"}]}},
        {"type": "bulleted_list_item",  "bulleted_list_item":  {"rich_text": [{"plain_text": "Bullet point"}]}},
        {"type": "numbered_list_item",  "numbered_list_item":  {"rich_text": [{"plain_text": "First item"}]}},
        {"type": "quote",               "quote":               {"rich_text": [{"plain_text": "A quote"}]}},
        {"type": "callout",             "callout":             {"rich_text": [{"plain_text": "Note"}], "icon": {"type": "emoji", "emoji": "💡"}}},
        {"type": "toggle",              "toggle":              {"rich_text": [{"plain_text": "Toggle text"}]}},
        {"type": "to_do",               "to_do":               {"rich_text": [{"plain_text": "Do this"}], "checked": False}},
        {"type": "code",                "code":                {"rich_text": [{"plain_text": "print('hi')"}], "language": "python"}},
        {"type": "table_row",           "table_row":           {"cells": [[{"plain_text": "Col A"}], [{"plain_text": "Col B"}]]}},
        {"type": "child_page",          "child_page":          {"title": "Child page name"}},
        {"type": "child_database",      "child_database":      {"title": "Child DB name"}},
        {"type": "image",               "image":               {}},  # unsupported — should return ""
    ]

    expected = [
        "Hello world",
        "# Title",
        "## Subtitle",
        "### Section",
        "• Bullet point",
        "1. First item",
        "> A quote",
        "💡 Note",
        "Toggle text",
        "[ ] Do this",
        "```python\nprint('hi')\n```",
        "Col A | Col B",
        "[Page: Child page name]",
        "[Database: Child DB name]",
        "",  # unsupported image block
    ]

    all_passed = True
    for mock, exp in zip(mock_blocks, expected):
        result = parser.parse_block(mock)
        status = "✓" if result == exp else "✗"
        if result != exp:
            all_passed = False
            print(f"  {status} [{mock['type']}]  expected='{exp}'  got='{result}'")
        else:
            print(f"  {status} [{mock['type']}]  '{result[:50]}'")

    assert all_passed, "Some block types parsed incorrectly — see above"

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    main()