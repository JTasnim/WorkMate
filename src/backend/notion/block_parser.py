class BlockParser:
    """
    Converts Notion block objects into plain text strings.
    Each block type has its own handler. Unsupported types return empty string.
    """

    def parse_block(self, block: dict) -> str:
        """
        Dispatch a block to its type-specific parser.
        Returns a plain text string. Never raises — unsupported types return ''.
        """
        block_type = block.get("type", "")
        handler = getattr(self, f"_parse_{block_type}", self._parse_unsupported)
        return handler(block)

    # ─────────────────────────────────────────
    # Internal helper
    # ─────────────────────────────────────────

    def _extract_rich_text(self, rich_text_array: list) -> str:
        """
        Notion stores all text as an array of rich_text objects.
        Each object has a plain_text field — concatenate them all.
        """
        return "".join(item.get("plain_text", "") for item in rich_text_array)

    def _parse_unsupported(self, block: dict) -> str:
        return ""

    # ─────────────────────────────────────────
    # Block type handlers
    # ─────────────────────────────────────────

    def _parse_paragraph(self, block: dict) -> str:
        text = self._extract_rich_text(block["paragraph"]["rich_text"])
        return text

    def _parse_heading_1(self, block: dict) -> str:
        text = self._extract_rich_text(block["heading_1"]["rich_text"])
        return f"# {text}"

    def _parse_heading_2(self, block: dict) -> str:
        text = self._extract_rich_text(block["heading_2"]["rich_text"])
        return f"## {text}"

    def _parse_heading_3(self, block: dict) -> str:
        text = self._extract_rich_text(block["heading_3"]["rich_text"])
        return f"### {text}"

    def _parse_bulleted_list_item(self, block: dict) -> str:
        text = self._extract_rich_text(block["bulleted_list_item"]["rich_text"])
        return f"• {text}"

    def _parse_numbered_list_item(self, block: dict) -> str:
        text = self._extract_rich_text(block["numbered_list_item"]["rich_text"])
        return f"1. {text}"

    def _parse_quote(self, block: dict) -> str:
        text = self._extract_rich_text(block["quote"]["rich_text"])
        return f"> {text}"

    def _parse_callout(self, block: dict) -> str:
        text = self._extract_rich_text(block["callout"]["rich_text"])
        icon = block["callout"].get("icon", {})
        emoji = icon.get("emoji", "→") if icon.get("type") == "emoji" else "→"
        return f"{emoji} {text}"

    def _parse_toggle(self, block: dict) -> str:
        text = self._extract_rich_text(block["toggle"]["rich_text"])
        return text

    def _parse_to_do(self, block: dict) -> str:
        text = self._extract_rich_text(block["to_do"]["rich_text"])
        checked = block["to_do"].get("checked", False)
        prefix = "[x]" if checked else "[ ]"
        return f"{prefix} {text}"

    def _parse_code(self, block: dict) -> str:
        text = self._extract_rich_text(block["code"]["rich_text"])
        language = block["code"].get("language", "")
        return f"```{language}\n{text}\n```"

    def _parse_table_row(self, block: dict) -> str:
        cells = block["table_row"]["cells"]
        cell_texts = [self._extract_rich_text(cell) for cell in cells]
        return " | ".join(cell_texts)

    def _parse_child_page(self, block: dict) -> str:
        title = block["child_page"].get("title", "")
        return f"[Page: {title}]"

    def _parse_child_database(self, block: dict) -> str:
        title = block["child_database"].get("title", "")
        return f"[Database: {title}]"