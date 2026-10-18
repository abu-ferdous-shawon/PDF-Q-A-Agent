from dataclasses import dataclass, field


@dataclass
class DocumentInfo:

    document_id: str

    source: str

    title: str

    document_type: str

    language: str

    total_pages: int

    parent_ids: list = field(default_factory=list)

    child_chunks: int = 0

    front_matter: dict = field(default_factory=dict)