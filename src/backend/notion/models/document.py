from dataclasses import dataclass, field


@dataclass
class NotionDocument:
    id: str
    title: str
    content: str
    parent_id: str
    url: str
    source_type: str = "page"
    properties: dict = field(default_factory=dict)
    created_time: str = ""
    last_edited_time: str = ""

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "title":            self.title,
            "content":          self.content,
            "parent_id":        self.parent_id,
            "url":              self.url,
            "source_type":      self.source_type,
            "properties":       self.properties,
            "created_time":     self.created_time,
            "last_edited_time": self.last_edited_time,
        }