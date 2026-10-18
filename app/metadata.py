import re
import os

import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def detect_document_type(text: str):

    text = text.lower()

    if "abstract" in text and "references" in text:
        return "research"

    if "thesis" in text:
        return "thesis"

    if "curriculum vitae" in text:
        return "cv"

    if "resume" in text:
        return "cv"

    if "meeting minutes" in text:
        return "minutes"

    if "invoice" in text:
        return "invoice"

    if "chapter" in text:
        return "book"

    if "table of contents" in text:
        return "book"

    if "policy" in text:
        return "policy"

    return "general"


def detect_language(text: str):

    english = sum(c.isascii() for c in text)

    ratio = english / max(len(text), 1)

    if ratio > 0.9:
        return "english"

    return "other"

def detect_title(docs):

    candidates = []

    for doc in docs[:2]:

        lines = [
            l.strip()
            for l in doc.page_content.split("\n")
            if l.strip()
        ]

        candidates.extend(lines[:8])

    for line in candidates:

        if len(line) < 5:
            continue

        if len(line) > 120:
            continue

        if line.lower().startswith("page"):
            continue

        if re.match(r"^\d+$", line):
            continue

        return line

    return "Unknown"


FRONT_MATTER_PROMPT = ChatPromptTemplate.from_template("""
You are extracting metadata from the first page(s) of a document.
The document could be ANY type -- thesis, contract, invoice, report,
letter, manual, resume, etc. Do not assume a fixed schema.

Read the text below and extract any clearly labeled metadata fields
that are explicitly present, such as (but not limited to): document
title, author(s), supervisor/advisor, institution, department, date,
recipient, sender, vendor, invoice number, contract parties, or any
other clearly identifiable named person, role, or field relevant to
THIS document.

Rules:
- Only extract information explicitly stated in the text.
- If a label and its value are separated (e.g. a name appears several
  lines below its role heading, such as in a signature block), still
  associate them correctly based on document structure and reading
  order.
- Do not guess or infer anything not stated.
- Return ONLY a JSON object mapping short snake_case field names to
  plain string values. If nothing is extractable, return {{}}.

Text:
---
{text}
---

JSON:
""")


def extract_front_matter_llm(llm, text: str) -> dict:
    """
    Runs once per document at ingest time (cheap -- first-page text
    only). Returns a dict of whatever metadata fields the LLM found,
    with keys it chose itself -- not a fixed schema, so this works
    across arbitrary document types.
    """
    raw = (FRONT_MATTER_PROMPT | llm | StrOutputParser()).invoke({"text": text[:6000]})

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        pass

    return {}


def extract_section(text: str) -> str:
    """
    Returns the first '## Heading' marker found in this chunk (added
    by docling_loader.py), or 'body' if the chunk starts mid-section
    with no heading of its own. Works for ANY document's own heading
    vocabulary -- no fixed keyword list required.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            return line[3:].strip()
    return "body"