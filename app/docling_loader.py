
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling.datamodel.base_models import InputFormat

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)

from langchain_core.documents import Document


# ==========================================================
# STEP 1: PDF PIPELINE SETTINGS
# ==========================================================

pipeline_options = PdfPipelineOptions()

# Disable OCR and image processing
pipeline_options.do_ocr = False
pipeline_options.generate_picture_images = False
pipeline_options.generate_page_images = False

# Keep structured table extraction
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

# Disable formula enrichment
pipeline_options.do_formula_enrichment = False

# CPU processing
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.CPU,
)


# ==========================================================
# STEP 2: INITIALIZE CONVERTER
# ==========================================================

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)


# ==========================================================
# STEP 3: HEADING LABELS
# ==========================================================

HEADING_LABELS = {
    "section_header",
    "title",
}


# ==========================================================
# STEP 4: LOAD PDF
# ==========================================================

def load_pdf(path):
    """
    Extract text-only content from PDF files.

    Features:
    - Extract selectable PDF text
    - Preserve document headings
    - Extract structured tables
    - Preserve page numbers
    - Build document outline
    - Ignore all images
    - Disable OCR

    Returns:
        langchain_docs
        outline
    """

    print(f"\nLoading PDF: {path}")

    result = converter.convert(path)

    doc = result.document

    pages = {}
    outline = []

    # ======================================================
    # STEP 5: EXTRACT DOCUMENT ITEMS
    # ======================================================

    for item, _level in doc.iterate_items():

        item_type = type(item).__name__

        # Ignore images completely
        if item_type == "PictureItem":
            continue

        prov = getattr(item, "prov", None)

        if not prov:
            continue

        page_no = prov[0].page_no

        pages.setdefault(page_no, [])

        label = str(
            getattr(item, "label", "")
        ).lower()

        # --------------------------------------------------
        # TABLE EXTRACTION
        # --------------------------------------------------

        if item_type == "TableItem":

            try:
                table_md = item.export_to_markdown(doc)
            except Exception:
                table_md = ""

            if table_md.strip():
                pages[page_no].append(
                    ("table", table_md)
                )

            continue

        # --------------------------------------------------
        # TEXT EXTRACTION
        # --------------------------------------------------

        text = getattr(item, "text", None)

        if not text or not text.strip():
            continue

        text = text.strip()

        # --------------------------------------------------
        # HEADING DETECTION
        # --------------------------------------------------

        is_heading = any(
            heading in label
            for heading in HEADING_LABELS
        )

        if is_heading:

            pages[page_no].append(
                ("text", f"## {text}")
            )

            outline.append({
                "page": page_no - 1,
                "heading": text,
            })

        else:

            pages[page_no].append(
                ("text", text)
            )

    # ======================================================
    # STEP 6: VALIDATE EXTRACTION
    # ======================================================

    if not pages:
        raise ValueError(
            f"No selectable text or tables extracted from: {path}"
        )

    # ======================================================
    # STEP 7: CREATE LANGCHAIN DOCUMENTS
    # ======================================================

    langchain_docs = []

    total_pages = max(pages.keys())

    for page_no in range(1, total_pages + 1):

        pieces = pages.get(page_no, [])

        page_text = "\n\n".join(
            content
            for _, content in pieces
        )

        has_table = any(
            kind == "table"
            for kind, _ in pieces
        )

        if not pieces:
            content_type = "empty"

        elif has_table:
            content_type = "table"

        else:
            content_type = "text"

        langchain_docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "page": page_no - 1,
                    "content_type": content_type,
                },
            )
        )

    # ======================================================
    # STEP 8: REPORT
    # ======================================================

    print(
        f"Extracted {len(langchain_docs)} PDF pages."
    )

    print(
        f"Detected {len(outline)} section headings."
    )

    print("Image extraction: Disabled")
    print("OCR: Disabled")

    return langchain_docs, outline
