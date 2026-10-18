import contextlib
import io
import logging
import os
import pickle
import re
import warnings

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore', category=FutureWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from hybrid_retriever import HybridRetriever
from utils import format_docs
from reranker import rerank
from compressor import compress
from memory import ConversationMemory
from query_rewriter import rewrite_question
from hyde import generate_hypothetical_answer
from judge import has_enough_context
from verifier import verify_answer
from classifier import classify_question
from confidence import calculate_confidence
from document_selector import DocumentSelector

DEBUG = False  # Set True to show diagnostic errors only.
UNKNOWN = "I don't know based on the provided documents."
SHORT_DOC_PAGE_THRESHOLD = 6

@contextlib.contextmanager
def quiet_dependencies():
    """Hide routine stdout/stderr produced by RAG libraries during a demo."""
    if DEBUG:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

with quiet_dependencies():
    with open('parent_store.pkl', 'rb') as f:
        parent_store = pickle.load(f)
    with open('outline_index.pkl', 'rb') as f:
        outline_index = pickle.load(f)
    with open('document_registry.pkl', 'rb') as f:
        document_registry = pickle.load(f)
    with open('chunks.pkl', 'rb') as f:
        documents = pickle.load(f)
    with open('front_matter_index.pkl', 'rb') as f:
        front_matter_index = pickle.load(f)

    memory = ConversationMemory(max_turns=5)
    selector = DocumentSelector(document_registry)
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embedding)
    llm = ChatGoogleGenerativeAI(model='gemini-flash-lite-latest', temperature=0)
    fast_retriever = HybridRetriever(vectorstore=vectorstore, llm=llm, k=20, fetch_k=40, all_documents=documents)
    deep_retriever = HybridRetriever(vectorstore=vectorstore, llm=llm, k=40, fetch_k=100, all_documents=documents)

prompt = ChatPromptTemplate.from_template('''
You are a document-grounded assistant. Answer only from the supplied context.
Do not treat models or findings described in a literature review as models
used or findings produced by the document's authors. When asked what models
were used, distinguish models evaluated from those selected for the final system.
Do not guess names, relationships, numbers, or facts.
Cite PDF pages as (Page N) only when a page citation appears in the context.
If evidence is absent, say exactly: I don't know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:
''')

# These are intent aliases, NOT guessed document metadata values.
FIELD_ALIASES = {
    'supervisor': ('supervisor', 'main_supervisor', 'thesis_supervisor', 'research_supervisor'),
    'co_supervisor': ('co_supervisor', 'cosupervisor', 'co_advisor', 'coadvisor'),
    'author': ('author', 'authors', 'student_name', 'student', 'prepared_by'),
    'title': ('title', 'document_title', 'thesis_title'),
    'advisor': ('advisor', 'academic_advisor'),
}


def requested_field(question):
    q = question.lower().replace('_', ' ').replace('-', ' ')
    if re.search(r'\b(co supervisor|cosupervisor|co advisor|coadvisor)\b', q):
        return 'co_supervisor'
    if re.search(r'\bsupervisor\b', q):
        return 'supervisor'
    if re.search(r'\badvisor\b', q):
        return 'advisor'
    if re.search(r'\b(author|authors|written by)\b', q):
        return 'author'
    if re.search(r'\b(title of|document title|thesis title)\b', q):
        return 'title'
    return None


def front_matter_context(question, selected_documents):
    """Only use an explicitly requested metadata key, never fuzzy co-supervisor matches."""
    field = requested_field(question)
    if not field:
        return None
    ids = selected_documents or list(document_registry)
    for doc_id in ids:
        fm = document_registry[doc_id].front_matter or {}
        normalized = {str(k).lower().replace('-', '_').replace(' ', '_'): v for k, v in fm.items()}
        for alias in FIELD_ALIASES[field]:
            value = normalized.get(alias)
            if value is not None and str(value).strip() and str(value).strip().lower() not in ('none', 'unknown', 'n/a', 'null'):
                return f"Document: {document_registry[doc_id].source}\n{field.replace('_', ' ')}: {value}"
    return None


def front_pages_context(selected_documents, max_pages=5):
    """Fallback to actual PDF text when metadata extraction omitted a person."""
    ids = selected_documents or list(document_registry)
    pages = [p for p in parent_store.values()
             if p.metadata.get('document_id') in ids
             and 0 <= p.metadata.get('page', -1) < max_pages]
    pages.sort(key=lambda p: (p.metadata.get('document_id', ''), p.metadata.get('page', 0)))
    return format_docs(pages)


def heading_matches(question, selected_documents, top_k=3):
    """Match exact topic words first, then semantic headings; selected PDFs only."""
    import numpy as np
    ids = selected_documents or list(outline_index)
    entries = []
    for doc_id in ids:
        entry = outline_index.get(doc_id, {})
        for i, heading in enumerate(entry.get('headings', [])):
            entries.append((doc_id, heading, entry['pages'][i], entry['embeddings'][i]))
    if not entries:
        return []
    q = question.lower()
    # Only boost exact meaningful heading topics, not generic words.
    topics = ('abstract', 'conclusion', 'methodology', 'methods', 'results',
              'limitations', 'recommendations', 'motivation', 'objectives',
              'contributions', 'discussion', 'summary of findings')
    requested = [t for t in topics if re.search(r'\b' + re.escape(t) + r'\b', q)]
    vec = np.asarray(embedding.embed_query(question))
    scored = []
    for doc_id, heading, page, emb in entries:
        v = np.asarray(emb)
        denom = np.linalg.norm(vec) * np.linalg.norm(v)
        similarity = float(np.dot(vec, v) / denom) if denom else 0.0
        clean = re.sub(r'^\s*(?:\d+\.)*\d+\s*', '', heading.lower()).strip()
        exact = any(t == clean or t in clean.split(' / ') for t in requested)
        related = any(t in clean for t in requested)
        score = similarity + (0.55 if exact else 0.22 if related else 0)
        if similarity >= 0.40 or exact or related:
            scored.append((score, doc_id, heading, page))
    scored.sort(reverse=True)
    return [(doc_id, heading, page) for _, doc_id, heading, page in scored[:top_k]]


def section_context(matches):
    seen = set()
    selected = []
    for doc_id, heading, page in matches:
        for parent in parent_store.values():
            md = parent.metadata
            if md.get('document_id') != doc_id:
                continue
            if md.get('page') == page or md.get('section', '').casefold() == heading.casefold():
                pid = md.get('parent_id')
                if pid not in seen:
                    seen.add(pid)
                    selected.append(parent)
    selected.sort(key=lambda p: (p.metadata.get('document_id', ''), p.metadata.get('page', 0)))
    return format_docs(selected)


def merge_unique(doc_lists):
    seen, merged = set(), []
    for group in doc_lists:
        for doc in group:
            key = (doc.metadata.get('document_id'), doc.metadata.get('page'), doc.page_content)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


def retrieve_context(retriever, question, hyde_query, selected_documents):
    docs = merge_unique([
        retriever.invoke(question, selected_documents),
        retriever.invoke(hyde_query, selected_documents),
    ])
    if not docs:
        return ''
    ranked = rerank(question, docs, top_k=15)
    seen, parents = set(), []
    for child in ranked:
        pid = child.metadata.get('parent_id')
        if pid not in seen and pid in parent_store:
            seen.add(pid)
            parents.append(parent_store[pid])
    if not parents:
        return ''
    compressed = compress(question, parents)
    return format_docs(compressed)


def answer_from(context, question):
    if not context.strip():
        return UNKNOWN
    return (prompt | llm | StrOutputParser()).invoke({
        'context': context, 'question': question
    }).strip()


def answer_document(question):
    history = memory.format()
    rewritten = rewrite_question(llm, history, question)
    selected = selector.select(rewritten)

    # For an explicitly selected short document, the entire PDF is affordable.
    if selected and len(selected) == 1 and document_registry[selected[0]].total_pages <= SHORT_DOC_PAGE_THRESHOLD:
        pages = [p for p in parent_store.values() if p.metadata.get('document_id') == selected[0]]
        pages.sort(key=lambda p: p.metadata.get('page', 0))
        return answer_from(format_docs(pages), rewritten)

    # Prefer exact field names; supervisor != co-supervisor.
    field = requested_field(rewritten)
    if field:
        context = front_matter_context(rewritten, selected)
        if context:
            result = answer_from(context, rewritten)
            if result != UNKNOWN:
                return result
        # Metadata can be incomplete. Consult the PDF's front pages instead.
        context = front_pages_context(selected)
        result = answer_from(context, rewritten)
        if result != UNKNOWN:
            return result
        # Do not stop here; allow regular retrieval to find the fact elsewhere.

    matches = heading_matches(rewritten, selected)
    if matches:
        context = section_context(matches)
        if context and has_enough_context(llm, rewritten, context):
            result = answer_from(context, rewritten)
            if result != UNKNOWN:
                return result

    hyde = generate_hypothetical_answer(llm, rewritten)
    for retriever in (fast_retriever, deep_retriever):
        context = retrieve_context(retriever, rewritten, hyde, selected)
        if context and has_enough_context(llm, rewritten, context):
            result = answer_from(context, rewritten)
            if result != UNKNOWN:
                if not verify_answer(llm, context, rewritten, result):
                    result = answer_from(context, rewritten + '\nOnly include directly supported facts.')
                return result
    return UNKNOWN


def main():
    while True:
        try:
            question = input('\nAsk Your Question: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ('exit', 'quit'):
            break
        try:
            with quiet_dependencies():
                question_type = classify_question(llm, question)
                if question_type == 'GREETING':
                    answer = 'Hello! What would you like to know?'
                elif question_type == 'CHAT':
                    answer = (llm | StrOutputParser()).invoke(question).strip()
                else:
                    answer = answer_document(question)
            print(f'\n{answer}\n')
            if question_type not in ('GREETING', 'CHAT'):
                memory.add(question, answer)
        except Exception as exc:
            if DEBUG:
                print(f'\nError: {exc}\n')
            else:
                print('\nSorry, I could not process that question. Please try again.\n')


if __name__ == '__main__':
    main()
















# from dotenv import load_dotenv
# import logging
# import pickle

# load_dotenv()

# # ----------------------------
# # LangChain Imports
# # ----------------------------
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from hybrid_retriever import HybridRetriever
# from utils import format_docs
# from reranker import rerank
# from compressor import compress
# from memory import ConversationMemory
# from query_rewriter import rewrite_question
# from hyde import generate_hypothetical_answer
# from judge import has_enough_context
# from verifier import verify_answer
# from classifier import classify_question
# from evaluator import log_retrieval
# from confidence import calculate_confidence
# from document_selector import DocumentSelector

# # ----------------------------
# # Enable MultiQuery Logs
# # ----------------------------
# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# # ----------------------------
# # Load persisted state
# # ----------------------------
# try:
#     with open("parent_store.pkl", "rb") as f:
#         parent_store = pickle.load(f)

#     with open("outline_index.pkl", "rb") as f:
#         outline_index = pickle.load(f)

#     with open("document_registry.pkl", "rb") as f:
#         document_registry = pickle.load(f)

#     with open("chunks.pkl", "rb") as f:
#         documents = pickle.load(f)
# except FileNotFoundError as e:
#     raise SystemExit(
#         f"Missing required file: {e.filename}. "
#         f"Run ingest.py before starting chat.py."
#     )

# memory = ConversationMemory(max_turns=5)
# selector = DocumentSelector(document_registry)

# # ----------------------------
# # Embedding Model
# # ----------------------------
# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # ----------------------------
# # Vector Database
# # ----------------------------
# vectorstore = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embedding,
# )

# # ----------------------------
# # Gemini LLM
# # ----------------------------
# llm = ChatGoogleGenerativeAI(
#     model="gemini-flash-lite-latest",
#     temperature=0,
# )

# # ----------------------------
# # Retrievers (document_id pre-filtered internally per query)
# # ----------------------------
# fast_retriever = HybridRetriever(
#     vectorstore=vectorstore, llm=llm, k=20, fetch_k=40,
#     all_documents=documents,
# )
# deep_retriever = HybridRetriever(
#     vectorstore=vectorstore, llm=llm, k=40, fetch_k=100,
#     all_documents=documents,
# )

# # ----------------------------
# # Section fast-path (dynamic heading match, any document type)
# # ----------------------------
# import numpy as np


# def find_best_heading_matches(question, selected_documents, outline_index, embedding,
#                                 threshold=0.40, top_k=3):
#     """
#     Returns up to top_k (document_id, heading, page, score) tuples
#     above threshold, sorted by score descending.

#     Multiple matches are returned (not just the single best) because
#     a document can have several headings on the same topic with
#     different scope -- e.g. "Margin of Error" (definition) vs.
#     "What Affects the Margin of Error?" (factors) -- which share
#     almost all their words, so pure text-similarity can pick the
#     wrong one. Gathering a few candidates and letting
#     has_enough_context judge the combined content is more reliable
#     than trusting a single top-1 match.
#     """
#     doc_ids = selected_documents if selected_documents else list(outline_index.keys())

#     all_headings, all_pages, all_doc_ids, all_embeddings = [], [], [], []
#     for doc_id in doc_ids:
#         entry = outline_index.get(doc_id)
#         if not entry or len(entry["headings"]) == 0:
#             continue
#         all_headings.extend(entry["headings"])
#         all_pages.extend(entry["pages"])
#         all_doc_ids.extend([doc_id] * len(entry["headings"]))
#         all_embeddings.append(entry["embeddings"])

#     if not all_embeddings:
#         return []

#     heading_matrix = np.vstack(all_embeddings)
#     question_vec = np.array(embedding.embed_query(question))

#     norms = np.linalg.norm(heading_matrix, axis=1) * np.linalg.norm(question_vec)
#     norms[norms == 0] = 1e-8
#     sims = heading_matrix @ question_vec / norms

#     ranked_idx = np.argsort(sims)[::-1]

#     matches = []
#     for idx in ranked_idx[:top_k]:
#         score = float(sims[idx])
#         if score < threshold:
#             break
#         matches.append((all_doc_ids[idx], all_headings[idx], all_pages[idx], score))

#     return matches




# def get_heading_match_chunks(heading_matches, parent_store):

#     chunks = []
#     seen_parent_ids = set()

#     for doc_id, heading, page, score in heading_matches:

#         for parent in parent_store.values():

#             metadata = parent.metadata

#             if metadata.get("document_id") != doc_id:
#                 continue

#             parent_page = metadata.get("page", -1)
#             parent_section = metadata.get("section", "")

#             # Prefer chunks explicitly labeled with the heading.
#             # Also include the page where the outline says
#             # the section begins.
#             if (
#                 parent_section == heading
#                 or parent_page == page
#             ):

#                 parent_id = metadata.get("parent_id")

#                 if parent_id not in seen_parent_ids:
#                     seen_parent_ids.add(parent_id)
#                     chunks.append(parent)

#     return chunks


# SHORT_DOC_PAGE_THRESHOLD = 6  # pages -- tune based on your typical doc sizes


# def get_full_document_context(document_id, parent_store):
#     """
#     Returns the full document's text, in page order, by concatenating
#     every parent chunk belonging to it. Only sensible for short
#     documents -- for anything long this would blow past context
#     limits and defeats the point of retrieval.
#     """
#     chunks = [
#         p for p in parent_store.values()
#         if p.metadata.get("document_id") == document_id
#     ]
#     chunks.sort(key=lambda d: d.metadata.get("page", 0))
#     return format_docs(chunks)


# with open("front_matter_index.pkl", "rb") as f:
#     front_matter_index = pickle.load(f)


# def find_best_front_matter_match(question, selected_documents, front_matter_index, embedding, threshold=0.45):
#     doc_ids = selected_documents if selected_documents else list(front_matter_index.keys())

#     all_keys, all_values, all_doc_ids, all_embeddings = [], [], [], []
#     for doc_id in doc_ids:
#         entry = front_matter_index.get(doc_id)
#         if not entry or len(entry["keys"]) == 0:
#             continue
#         all_keys.extend(entry["keys"])
#         all_values.extend(entry["values"])
#         all_doc_ids.extend([doc_id] * len(entry["keys"]))
#         all_embeddings.append(entry["embeddings"])

#     if not all_embeddings:
#         return None

#     matrix = np.vstack(all_embeddings)
#     qvec = np.array(embedding.embed_query(question))

#     norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(qvec)
#     norms[norms == 0] = 1e-8
#     sims = matrix @ qvec / norms

#     best_idx = int(np.argmax(sims))
#     best_score = float(sims[best_idx])

#     if best_score < threshold:
#         return None

#     return all_doc_ids[best_idx], all_keys[best_idx], all_values[best_idx], best_score


# # ----------------------------
# # Prompt
# # ----------------------------
# prompt = ChatPromptTemplate.from_template("""
# You are an AI assistant.

# Answer ONLY using the provided context.

# When you use information from the context,
# cite the page number inline like this:

# (Page 47)

# At the end, include a short Sources section listing all unique page numbers.

# If the answer is not present, reply exactly:

# "I don't know based on the provided documents."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)


# def merge_unique(doc_lists):
#     seen = set()
#     merged = []
#     for docs in doc_lists:
#         for doc in docs:
#             key = (
#                 doc.metadata["document_id"],
#                 doc.metadata["page"],
#                 doc.page_content,
#             )
#             if key in seen:
#                 continue
#             seen.add(key)
#             merged.append(doc)
#     return merged


# def group_by_parent(child_docs, parent_store):
#     seen = set()
#     parent_docs = []
#     for child in child_docs:
#         parent_id = child.metadata["parent_id"]
#         if parent_id not in seen:
#             seen.add(parent_id)
#             parent_docs.append(parent_store[parent_id])
#     return parent_docs


# def run_retrieval_pass(retriever, question, hyde_query, selected_documents):
#     """One retrieve -> rerank -> group -> compress pass. Returns
#     (child_docs, parent_docs, compressed_docs, context)."""
#     docs1 = retriever.invoke(question, selected_documents)
#     docs2 = retriever.invoke(hyde_query, selected_documents)

#     child_docs = merge_unique([docs1, docs2])
#     print(f"Retrieved {len(child_docs)} unique chunks.")

#     child_docs = rerank(
#     question,
#     child_docs,
#     top_k=15,
# )
#     parent_docs = group_by_parent(child_docs, parent_store)

#     print("\n================ PARENT DOCS ================\n")
#     for doc in parent_docs:
#         print("=" * 80)
#         print("Page:", doc.metadata.get("page"))
#         print(doc.page_content[:500])
#         print()

#     compressed_docs = compress(question, parent_docs)

#     print("\n================ COMPRESSED DOCS ================\n")
#     for doc in compressed_docs:
#         print("=" * 80)
#         print("Page:", doc.metadata.get("page"))
#         print(doc.page_content[:1200])
#         print()

#     context = format_docs(compressed_docs)
#     return child_docs, parent_docs, compressed_docs, context


# # ----------------------------
# # Chat Loop
# # ----------------------------
# while True:

#     question = input("\nAsk: ")

#     if question.lower() == "exit":
#         break

#     question_type = classify_question(llm, question)
#     print("Question Type:", question_type)

#     if question_type == "GREETING":
#         print("\nHello! How can I help you today?")
#         continue

#     if question_type == "CHAT":
#         try:
#             answer = (llm | StrOutputParser()).invoke(question)
#         except Exception as e:
#             answer = "Sorry, I ran into an error processing that. Please try again."
#             print(f"[error] CHAT branch: {e}")
#         print("\nAnswer:\n")
#         print(answer)
#         continue

#     # question_type == "DOCUMENT"
#     try:
#         history = memory.format()
#         rewritten_question = rewrite_question(llm, history, question)

#         print("\nRewritten Query:")
#         print(rewritten_question)

#         selected_documents = selector.select(rewritten_question)

#         if selected_documents:
#             print("\nSelected Documents:")
#             for doc_id in selected_documents:
#                 print("-", document_registry[doc_id].source)
#         else:
#             print("\nSearching all documents.")

#         # -------------------------------------------------
#         # Short-document fast-path: skip retrieval entirely
#         # -------------------------------------------------
#         if selected_documents and len(selected_documents) == 1:
#             doc_id = selected_documents[0]
#             reg = document_registry[doc_id]

#             if reg.total_pages <= SHORT_DOC_PAGE_THRESHOLD:
#                 context = get_full_document_context(doc_id, parent_store)

#                 if context.strip():
#                     print(f"\nShort document ({reg.total_pages} page(s)) -- "
#                           f"using full document as context, skipping retrieval.")

#                     answer = (prompt | llm | StrOutputParser()).invoke(
#                         {"context": context, "question": rewritten_question}
#                     )

#                     verified = verify_answer(llm, context, rewritten_question, answer)
#                     print("\nVerification:", verified)

#                     if not verified:
#                         print("\nVerification failed. Regenerating conservatively...")
#                         answer = (prompt | llm | StrOutputParser()).invoke(
#                             {
#                                 "context": context,
#                                 "question": rewritten_question + """

# Be conservative.

# Do not infer anything.

# Only answer what is explicitly supported.
# """,
#                             }
#                         )

#                     print("\nAnswer:\n")
#                     print(answer)
#                     print("\nConfidence: 92%")

#                     memory.add(question, answer)
#                     continue

#                 print("\nShort document selected but no content found, "
#                       "falling back to normal retrieval.")

#         # -------------------------------------------------
#         # Front-matter fast-path (supervisor, author, etc.)
#         # -------------------------------------------------
#         front_matter_match = find_best_front_matter_match(
#             rewritten_question, selected_documents, front_matter_index, embedding
#         )

#         if front_matter_match:
#             doc_id, key, value, score = front_matter_match
#             print(f"\nFront-matter fast-path hit: '{key}' (score {score:.3f})")

#             context = f"{key.replace('_', ' ')}: {value}"

#             answer = (prompt | llm | StrOutputParser()).invoke(
#                 {"context": context, "question": rewritten_question}
#             )

#             print("\nAnswer:\n")
#             print(answer)
#             print("\nConfidence: 90%")

#             memory.add(question, answer)
#             continue

#         print("\nNo front-matter field matched confidently, "
#               "falling back to semantic retrieval.")

#         # -------------------------------------------------
#         # Section fast-path: dynamic heading match
#         # -------------------------------------------------
#         heading_matches = find_best_heading_matches(
#             rewritten_question, selected_documents, outline_index, embedding
#         )

#         if heading_matches:
#             print("\nHeading fast-path candidates:")
#             for doc_id, heading, page, score in heading_matches:
#                 print(f"  - '{heading}' (page {page + 1}, score {score:.3f})")

#             section_docs = get_heading_match_chunks(heading_matches, parent_store)

#             if section_docs:
#                 context = format_docs(section_docs)

#                 # Don't trust the fast-path blindly -- if the matched
#                 # heading(s) turn out not to actually contain enough
#                 # to answer the question (e.g. "what is X" matched a
#                 # "what affects X" section), fall through to full
#                 # retrieval instead of returning "I don't know".
#                 enough = has_enough_context(llm, rewritten_question, context)

#                 if enough:
#                     confidence = 90

#                     answer = (prompt | llm | StrOutputParser()).invoke(
#                         {"context": context, "question": rewritten_question}
#                     )

#                     print("\nAnswer:\n")
#                     print(answer)
#                     print(f"\nConfidence: {confidence}%")

#                     memory.add(question, answer)
#                     continue

#                 print("\nHeading fast-path content insufficient, "
#                       "falling back to semantic retrieval.")
#             else:
#                 print("\nNo parent chunks found for matched heading(s), "
#                       "falling back to semantic retrieval.")

#         # -------------------------------------------------
#         # Normal retrieval pipeline
#         # -------------------------------------------------
#         hyde_query = generate_hypothetical_answer(llm, rewritten_question)
#         print("\nHyDE Query:")
#         print(hyde_query)

#         child_docs, parent_docs, compressed_docs, context = run_retrieval_pass(
#             fast_retriever, rewritten_question, hyde_query, selected_documents
#         )

#         log_retrieval(
#             question=question,
#             rewritten_question=rewritten_question,
#             hyde_query=hyde_query,
#             child_docs=child_docs,
#             parent_docs=parent_docs,
#             compressed_docs=compressed_docs,
#         )

#         enough = has_enough_context(llm, rewritten_question, context)
#         confidence = calculate_confidence(child_docs, parent_docs, compressed_docs, enough)

#         if not enough:
#             print("\nInitial retrieval failed.")
#             print("Retrying with deep retriever...")

#             child_docs, parent_docs, compressed_docs, context = run_retrieval_pass(
#                 deep_retriever, rewritten_question, hyde_query, selected_documents
#             )

#             compressed_docs.sort(
#                 key=lambda d: (d.metadata.get("document_id", ""), d.metadata.get("page", 0))
#             )
#             context = format_docs(compressed_docs)

#             enough = has_enough_context(llm, rewritten_question, context)
#             print("Second Context Check:", enough)

#             confidence = calculate_confidence(child_docs, parent_docs, compressed_docs, enough)

#             if not enough:
#                 print("\nAnswer:\n")
#                 print("I don't know based on the provided documents.")
#                 memory.add(question, "I don't know based on the provided documents.")
#                 continue

#         print("\nContext Check:", enough)

#         answer = (prompt | llm | StrOutputParser()).invoke(
#             {"context": context, "question": rewritten_question}
#         )

#         verified = verify_answer(llm, context, rewritten_question, answer)
#         print("\nVerification:", verified)

#         if not verified:
#             print("\nVerification failed. Regenerating conservatively...")
#             answer = (prompt | llm | StrOutputParser()).invoke(
#                 {
#                     "context": context,
#                     "question": rewritten_question + """

# Be conservative.

# Do not infer anything.

# Only answer what is explicitly supported.
# """,
#                 }
#             )

#         print("\nAnswer:\n")
#         print(answer)
#         print(f"\nConfidence: {confidence}%")

#         memory.add(question, answer)

#     except Exception as e:
#         print(f"\n[error] Something went wrong processing that question: {e}")
#         print("Please try rephrasing, or type 'exit' to quit.")