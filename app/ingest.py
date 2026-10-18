from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import pickle
import uuid
import re
import hashlib

from docling_loader import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




from langchain_google_genai import ChatGoogleGenerativeAI

from metadata import (
    detect_document_type,
    detect_language,
    detect_title,
    extract_section,
    extract_front_matter_llm,   # <-- add this
)

from document_registry import DocumentInfo


# ====================================================
# SETTINGS
# ====================================================

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

DATA_FOLDER = "data"
DB_PATH = "chroma_db"
CACHE_DIR = "docling_cache"



EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RESET_DATABASE = True

os.makedirs(CACHE_DIR, exist_ok=True)


def file_hash(path):
    """Content hash -- cache invalidates automatically if a PDF's
    contents change, but survives renames/moves within data/."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ====================================================
# STEP 1: RESET DATABASE (OPTIONAL)
# ====================================================

if RESET_DATABASE and os.path.exists(DB_PATH):
    print("Deleting old vector database...")
    shutil.rmtree(DB_PATH)


# ====================================================
# STEP 2: LOAD ALL PDFS
# ====================================================
document_registry = {}
raw_outlines = {}

docs = []
total_pages = 0

pdf_files = sorted(
    f
    for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(".pdf")
)

print(f"\nFound {len(pdf_files)} PDF(s).\n")

for pdf in pdf_files:

    pdf_path = os.path.join(DATA_FOLDER, pdf)

    print(f"Loading {pdf}")

    
    # Include the text-only extraction version in the cache key.
    # This prevents reuse of old image-enriched content.
    file_key = file_hash(pdf_path)
    cache_path = os.path.join(
        CACHE_DIR,
        f"{file_key}_text_only_v1.pkl"
    )

    if os.path.exists(cache_path):

        print(
            "  [cache hit] Reusing previously "
            "extracted text-only content"
        )

        with open(cache_path, "rb") as f:
            pdf_docs, doc_outline = pickle.load(f)

    else:

        try:
            pdf_docs, doc_outline = load_pdf(pdf_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [skipped] Failed to load {pdf}: {e}")
            continue

        with open(cache_path, "wb") as f:
            pickle.dump(
                (pdf_docs, doc_outline),
                f
            )

    print(
        f"  Extracted {len(pdf_docs)} pages "
        f"and {len(doc_outline)} headings."
    )



    document_id = str(uuid.uuid4())

    # -------------------------
    # Detect document metadata
    # -------------------------

    document_text = "\n".join(
        doc.page_content
        for doc in pdf_docs[:5]
    )

    document_type = detect_document_type(document_text)

    language = detect_language(document_text)

    title = detect_title(pdf_docs)
    
    try:
        front_matter = extract_front_matter_llm(llm, document_text)
    except Exception as e:
        print(f"  [warn] Front-matter extraction failed: {e}")
        front_matter = {}

    # -------------------------
    # Create registry entry
    # -------------------------

    registry = DocumentInfo(
        document_id=document_id,
        source=pdf,
        title=title,
        document_type=document_type,
        language=language,
        total_pages=len(pdf_docs),
        front_matter=front_matter,
    )

    document_registry[document_id] = registry
    raw_outlines[document_id] = doc_outline
    print("\n  Detected headings:")

    for heading in doc_outline:
        print(
            f"    Page {heading['page'] + 1}: "
            f"{heading['heading']}"
        )
    total_pages += len(pdf_docs)

    for doc in pdf_docs:

        doc.metadata["document_id"] = document_id
        doc.metadata["source"] = pdf
        doc.metadata["file_name"] = pdf
        doc.metadata["file_type"] = "pdf"
        doc.metadata["title"] = title
        doc.metadata["language"] = language
        doc.metadata["document_type"] = document_type

    docs.extend(pdf_docs)
print(f"\nLoaded {total_pages} pages.")


empty_pages = [
    d for d in docs
    if d.metadata.get("content_type") == "empty"
]

if empty_pages:
    print(
        f"\nWarning: {len(empty_pages)} page(s) "
        "had no extractable text or tables. "
        "OCR is disabled."
    )

# ====================================================
# STEP 3: CLEAN TEXT
# ====================================================

print("Cleaning text...")

for doc in docs:

    text = doc.page_content

    # Preserve paragraphs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    doc.page_content = text.strip()


# ====================================================
# STEP 4: SPLITTERS
# ====================================================

parent_splitter = RecursiveCharacterTextSplitter(

    chunk_size=4000,

    chunk_overlap=400,

    separators=[
        "\n# ",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        " ",
        "",
    ],
)

child_splitter = RecursiveCharacterTextSplitter(

    chunk_size=700,

    chunk_overlap=150,

    separators=[
        "\n# ",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        " ",
        "",
    ],
)


# ====================================================
# STEP 5: CREATE PARENT & CHILD CHUNKS
# ====================================================

parent_documents = []
child_documents = []

for doc in docs:

    document_id = doc.metadata["document_id"]

    registry = document_registry[document_id]

    page = doc.metadata.get("page", -1)

    source = doc.metadata["source"]

    parents = parent_splitter.split_documents([doc])

    for parent in parents:

        parent_id = str(uuid.uuid4())

        parent.metadata.update(
            {
                "document_id": document_id,

                "parent_id": parent_id,

                "page": page,

                "source": source,

                "file_name": source,

                "file_type": "pdf",

                "chunk_type": "parent",

                "title": doc.metadata["title"],

                "language": doc.metadata["language"],

                "document_type": doc.metadata["document_type"],

                "section": extract_section(parent.page_content),

                "content_type": doc.metadata.get("content_type", "text"),
            }
        )

        parent_documents.append(parent)
        registry.parent_ids.append(parent_id)

        children = child_splitter.split_documents([parent])

        for child in children:

            child.metadata.update(
                {
                    "document_id": document_id,

                    "parent_id": parent_id,

                    "page": page,

                    "source": source,

                    "file_name": source,

                    "file_type": "pdf",

                    "chunk_type": "child",

                    "title": doc.metadata["title"],

                    "language": doc.metadata["language"],

                    "document_type": doc.metadata["document_type"],

                    "section": parent.metadata["section"],

                    "content_type": doc.metadata.get("content_type", "text"),
                }
            )

            child_documents.append(child)
            registry.child_chunks += 1

print(f"\nParent Chunks : {len(parent_documents)}")
print(f"Child Chunks  : {len(child_documents)}")


# ====================================================
# STEP 6: SAVE STORES
# ====================================================

parent_store = {
    doc.metadata["parent_id"]: doc
    for doc in parent_documents
}

with open("parent_store.pkl", "wb") as f:
    pickle.dump(parent_store, f)

# ====================================================
# STEP 6b: BUILD SECTION INDEX
# ====================================================
# section_index[document_id][section_label] -> list of parent_ids
from collections import defaultdict

section_index = defaultdict(lambda: defaultdict(list))

for parent in parent_documents:
    doc_id = parent.metadata["document_id"]
    section = parent.metadata.get("section", "body")
    section_index[doc_id][section].append(parent.metadata["parent_id"])

section_index = {k: dict(v) for k, v in section_index.items()}

with open("section_index.pkl", "wb") as f:
    pickle.dump(section_index, f)

print("Saved section_index.pkl")


with open(
    "document_registry.pkl",
    "wb",
) as f:

    pickle.dump(
        document_registry,
        f,
    )

with open("chunks.pkl", "wb") as f:
    pickle.dump(child_documents, f)

print("Saved parent_store.pkl")
print("Saved chunks.pkl")


# ====================================================
# STEP 7: LOAD EMBEDDINGS
# ====================================================

print("\nLoading embedding model...")

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ====================================================
# STEP 6c: BUILD DYNAMIC OUTLINE INDEX (with embeddings)
# ====================================================
print("\nBuilding outline index...")

import numpy as np

outline_index = {}

for document_id, headings in raw_outlines.items():
    if not headings:
        outline_index[document_id] = {
            "headings": [], "pages": [], "embeddings": np.array([])
        }
        continue

    heading_texts = [h["heading"] for h in headings]
    heading_embeddings = embedding.embed_documents(heading_texts)

    outline_index[document_id] = {
        "headings": heading_texts,
        "pages": [h["page"] for h in headings],
        "embeddings": np.array(heading_embeddings),
    }

with open("outline_index.pkl", "wb") as f:
    pickle.dump(outline_index, f)

print(f"Saved outline_index.pkl ({sum(len(v['headings']) for v in outline_index.values())} headings total)")


# ====================================================
# STEP 6d: BUILD FRONT-MATTER INDEX (dynamic, embedding-based)
# ====================================================
print("\nBuilding front-matter index...")

front_matter_index = {}

for document_id, registry in document_registry.items():
    fm = registry.front_matter
    if not fm:
        front_matter_index[document_id] = {"keys": [], "values": [], "embeddings": np.array([])}
        continue

    keys = list(fm.keys())
    values = [fm[k] for k in keys]
    key_embeddings = embedding.embed_documents([k.replace("_", " ") for k in keys])

    front_matter_index[document_id] = {
        "keys": keys,
        "values": values,
        "embeddings": np.array(key_embeddings),
    }

with open("front_matter_index.pkl", "wb") as f:
    pickle.dump(front_matter_index, f)

print(f"Saved front_matter_index.pkl "
      f"({sum(len(v['keys']) for v in front_matter_index.values())} fields total)")


# ====================================================
# STEP 8: CREATE CHROMA DATABASE
# ====================================================

print("\nCreating vector database...")

vectorstore = Chroma.from_documents(

    documents=child_documents,

    embedding=embedding,

    ids=[str(uuid.uuid4()) for _ in child_documents],

    persist_directory=DB_PATH,
)

print("Vector database created successfully!")


# ====================================================
# STEP 9: FINAL REPORT
# ====================================================

print("\n" + "=" * 60)
print("\nDocuments\n")

for doc in document_registry.values():

    print("-"*50)

    print("Title :", doc.title)

    print("Source:", doc.source)

    print("Type  :", doc.document_type)

    print("Pages :", doc.total_pages)

    print("Parents:", len(doc.parent_ids))

    print("Children:", doc.child_chunks)


print("=" * 60)

print(f"PDF Files       : {len(pdf_files)}")
print(f"Original Pages  : {total_pages}")
print(f"Parent Chunks   : {len(parent_documents)}")
print(f"Child Chunks    : {len(child_documents)}")
print(f"Stored Vectors  : {vectorstore._collection.count()}")

print("=" * 60)