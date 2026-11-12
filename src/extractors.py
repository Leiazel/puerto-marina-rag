"""
Multi-format document text extraction module for Club Atlético Puerto Marina RAG (v3.0).
Extracts and converts PDF, DOCX, legacy DOC, and Markdown/TXT documents into clean,
structured Markdown with hierarchical headings and preserved tables.
"""
import io
import re
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pypdf
import docx
import olefile


class ExtractionError(Exception):
    """Raised when document extraction fails due to format or corruption."""
    pass


def validate_file_magic_bytes(filename: str, file_bytes: bytes) -> None:
    """Validates binary signatures for PDF, DOCX, DOC, and ensures text formats contain no null bytes."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" and not file_bytes.startswith(b"%PDF-"):
        raise ExtractionError("Firma binaria corrupta o inválida. No es un documento PDF estándar.")
    if ext == ".docx" and not file_bytes.startswith(b"PK\x03\x04"):
        raise ExtractionError("Firma binaria inválida. No es un documento DOCX válido (ZIP/OOXML).")
    if ext == ".doc" and not file_bytes.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        raise ExtractionError("Firma binaria inválida. No es un documento OLE2 (.doc) válido.")
    if ext in (".md", ".txt") and b"\x00" in file_bytes[:4096]:
        raise ExtractionError("El archivo de texto contiene bytes nulos binarios no válidos.")


def extract_text_from_pdf(content_or_path: Union[str, Path, bytes]) -> Dict[str, Any]:
    """
    Extracts text from a PDF document page by page, formatting output as structured Markdown.
    Detects scanned PDFs with no extractable text layer.
    """
    try:
        if isinstance(content_or_path, (str, Path)):
            reader = pypdf.PdfReader(str(content_or_path))
        else:
            reader = pypdf.PdfReader(io.BytesIO(content_or_path))

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ExtractionError("El archivo PDF no contiene páginas.")

        extracted_pages = []
        total_chars = 0

        for page_idx, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                extracted_pages.append(f"## Página {page_idx}\n\n{page_text}")
                total_chars += len(page_text)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Error al procesar el archivo PDF: {e}") from e

    if total_chars < 30:
        raise ExtractionError(
            "El archivo PDF no contiene texto digital extraíble (posible documento escaneado como imagen "
            "sin capa OCR). El motor RAG requiere documentos con texto digital seleccionable."
        )

    full_markdown = "\n\n".join(extracted_pages)
    return {
        "text": full_markdown,
        "format": "pdf",
        "page_count": total_pages,
        "char_count": total_chars
    }


def extract_text_from_docx(content_or_path: Union[str, Path, bytes]) -> Dict[str, Any]:
    """
    Extracts text and tables from DOCX using python-docx.
    Maps Heading styles to Markdown headers (#, ##, ###) and converts tables into Markdown tables.
    """
    try:
        if isinstance(content_or_path, (str, Path)):
            doc = docx.Document(str(content_or_path))
        else:
            doc = docx.Document(io.BytesIO(content_or_path))
    except Exception as e:
        raise ExtractionError(f"Error al abrir el archivo DOCX: {e}") from e

    elements = []
    total_chars = 0

    # Process block elements in document order (paragraphs and tables)
    for block in doc.element.body:
        if block.tag.endswith("p"):  # Paragraph
            p = next((p_obj for p_obj in doc.paragraphs if p_obj._p == block), None)
            if not p:
                continue
            text = p.text.strip()
            if not text:
                continue

            style_name = (p.style.name or "").lower()
            if "heading 1" in style_name or "título 1" in style_name:
                elements.append(f"# {text}")
            elif "heading 2" in style_name or "título 2" in style_name:
                elements.append(f"## {text}")
            elif "heading 3" in style_name or "título 3" in style_name:
                elements.append(f"### {text}")
            elif "heading 4" in style_name or "título 4" in style_name:
                elements.append(f"#### {text}")
            elif "list bullet" in style_name:
                elements.append(f"- {text}")
            elif "list number" in style_name:
                elements.append(f"1. {text}")
            else:
                elements.append(text)
            total_chars += len(text)

        elif block.tag.endswith("tbl"):  # Table
            tbl = next((t_obj for t_obj in doc.tables if t_obj._tbl == block), None)
            if not tbl or not tbl.rows:
                continue

            table_lines = []
            header_processed = False
            for row in tbl.rows:
                cells = [c.text.strip().replace("\n", " ").replace("|", "\\|") for c in row.cells]
                if not any(cells):
                    continue
                row_str = "| " + " | ".join(cells) + " |"
                table_lines.append(row_str)
                total_chars += sum(len(c) for c in cells)

                if not header_processed:
                    sep = "| " + " | ".join(["---"] * len(cells)) + " |"
                    table_lines.append(sep)
                    header_processed = True

            if table_lines:
                elements.append("\n".join(table_lines))

    if not elements or total_chars < 15:
        raise ExtractionError("El archivo DOCX no contiene texto o contenido procesable.")

    full_markdown = "\n\n".join(elements)
    return {
        "text": full_markdown,
        "format": "docx",
        "char_count": total_chars
    }


def extract_text_from_doc(content_or_path: Union[str, Path, bytes]) -> Dict[str, Any]:
    """
    Extracts text from legacy Microsoft Word 97-2003 (.doc) binary files.
    Uses olefile to inspect the OLE2 Compound File Structure.
    First tries FIB / PlcPcd piece table decoding (Word 97-2003 standard).
    Falls back to structured stream string extraction if the piece table is irregular.
    """
    ole = None
    try:
        if isinstance(content_or_path, (str, Path)):
            if not olefile.isOleFile(str(content_or_path)):
                raise ExtractionError("El archivo no es un documento OLE2 estructurado válido (.doc).")
            ole = olefile.OleFileIO(str(content_or_path))
        else:
            stream_bytes = io.BytesIO(content_or_path)
            if not olefile.isOleFile(stream_bytes):
                raise ExtractionError("El archivo no es un documento OLE2 estructurado válido (.doc).")
            ole = olefile.OleFileIO(stream_bytes)

        if not ole.exists("WordDocument"):
            raise ExtractionError("El contenedor OLE no contiene el flujo estándar 'WordDocument'.")

        word_doc_stream = ole.openstream("WordDocument").read()

        # Attempt standard FIB parsing
        extracted_text = ""
        try:
            extracted_text = _parse_doc_fib_piece_table(ole, word_doc_stream)
        except Exception:
            extracted_text = ""

        # Fallback to high-confidence stream text extraction
        if not extracted_text or len(extracted_text.strip()) < 30:
            extracted_text = _extract_clean_strings_from_doc_stream(word_doc_stream)

        cleaned_text = extracted_text.strip()
        if len(cleaned_text) < 20:
            raise ExtractionError("No se pudo extraer texto coherente del archivo .doc legado.")

        # Post-process into clean Markdown paragraphs
        paragraphs = [p.strip() for p in cleaned_text.split("\r") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in cleaned_text.split("\n") if p.strip()]

        formatted_md = "\n\n".join(paragraphs)
        return {
            "text": formatted_md,
            "format": "doc",
            "char_count": len(formatted_md)
        }
    finally:
        if ole:
            ole.close()


def _parse_doc_fib_piece_table(ole: olefile.OleFileIO, word_doc: bytes) -> str:
    """
    Parses the WordDocument FIB and PlcPcd (Piece Table) inside 0Table or 1Table stream.
    Supports both compressed 8-bit cp1252 pieces and uncompressed UTF-16LE pieces.
    """
    if len(word_doc) < 0x01AA:
        return ""

    # Offset 0x000A: flags. Bit 9 (mask 0x0200) determines whether 1Table (1) or 0Table (0) is used
    flags = int.from_bytes(word_doc[0x000A:0x000C], byteorder="little")
    use_1table = bool(flags & 0x0200)
    table_name = "1Table" if use_1table else "0Table"

    if not ole.exists(table_name):
        return ""

    table_stream = ole.openstream(table_name).read()

    # In Word 97-2003, fcClx is at 0x01A2, lcbClx is at 0x01A6 in the FIB
    fc_clx = int.from_bytes(word_doc[0x01A2:0x01A6], byteorder="little")
    lcb_clx = int.from_bytes(word_doc[0x01A6:0x01AA], byteorder="little")

    if fc_clx + lcb_clx > len(table_stream) or lcb_clx <= 0:
        return ""

    clx = table_stream[fc_clx:fc_clx + lcb_clx]
    pos = 0

    # Skip PRC entries (type 0x01)
    while pos < len(clx) and clx[pos] == 0x01:
        pos += 1
        if pos + 2 > len(clx):
            return ""
        cb = int.from_bytes(clx[pos:pos + 2], byteorder="little")
        pos += 2 + cb

    # Now we must find Pcdt (type 0x02)
    if pos >= len(clx) or clx[pos] != 0x02:
        return ""
    pos += 1

    lcb_pcd = int.from_bytes(clx[pos:pos + 4], byteorder="little")
    pos += 4
    pcd_data = clx[pos:pos + lcb_pcd]

    # PlcPcd format: (n+1) CPs of 4 bytes each, followed by n PCDs of 8 bytes each
    # n = (lcb_pcd - 4) / (4 + 8) = (lcb_pcd - 4) / 12
    if len(pcd_data) < 16:
        return ""

    n = (len(pcd_data) - 4) // 12
    if n <= 0:
        return ""

    cps = [int.from_bytes(pcd_data[i * 4:(i + 1) * 4], byteorder="little") for i in range(n + 1)]
    pcds_offset = (n + 1) * 4

    text_fragments = []
    for i in range(n):
        cp_len = cps[i + 1] - cps[i]
        if cp_len <= 0:
            continue

        pcd_entry = pcd_data[pcds_offset + i * 8:pcds_offset + (i + 1) * 8]
        fc = int.from_bytes(pcd_entry[2:6], byteorder="little")

        # Bit 30 of fc determines compression: 1 = compressed cp1252, 0 = UTF-16LE
        is_compressed = bool(fc & 0x40000000)
        actual_fc = fc & 0x3FFFFFFF

        if is_compressed:
            byte_offset = actual_fc // 2
            raw_bytes = word_doc[byte_offset:byte_offset + cp_len]
            text = raw_bytes.decode("cp1252", errors="replace")
        else:
            byte_offset = actual_fc
            raw_bytes = word_doc[byte_offset:byte_offset + (cp_len * 2)]
            text = raw_bytes.decode("utf-16le", errors="replace")

        # Filter standard Word control characters
        text = text.replace("\x07", "\t").replace("\x0b", "\n").replace("\x0c", "\n")
        text_fragments.append(text)

    return "".join(text_fragments)


def _extract_clean_strings_from_doc_stream(stream: bytes) -> str:
    """
    Robust fallback scanner for text inside WordDocument stream:
    Extracts contiguous runs of UTF-16LE or Latin-1 characters.
    """
    # 1. Try UTF-16LE regex
    utf16_runs = re.findall(b'(?:[\x20-\x7E\xA0-\xFF]\x00|\x0D\x00|\x0A\x00|\x09\x00){4,}', stream)
    best_utf16 = ""
    if utf16_runs:
        decoded_runs = [r.decode("utf-16le", errors="ignore") for r in utf16_runs]
        best_utf16 = "\n".join(r.strip() for r in decoded_runs if len(r.strip()) > 15)

    # 2. Try ASCII / CP1252 regex
    ascii_runs = re.findall(b'[\x20-\x7E\xA0-\xFF\r\n\t]{6,}', stream)
    best_ascii = ""
    if ascii_runs:
        decoded_ascii = [r.decode("cp1252", errors="ignore") for r in ascii_runs]
        best_ascii = "\n".join(r.strip() for r in decoded_ascii if len(r.strip()) > 15)

    return best_utf16 if len(best_utf16) >= len(best_ascii) else best_ascii


def extract_document_content(
    file_bytes_or_path: Union[str, Path, bytes],
    filename: str
) -> Dict[str, Any]:
    """
    Universal dispatcher that inspects file extension and extracts structured Markdown.
    Supports .pdf, .docx, .doc, .md, and .txt.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        res = extract_text_from_pdf(file_bytes_or_path)
    elif ext == ".docx":
        res = extract_text_from_docx(file_bytes_or_path)
    elif ext == ".doc":
        res = extract_text_from_doc(file_bytes_or_path)
    elif ext in (".md", ".txt"):
        if isinstance(file_bytes_or_path, (str, Path)):
            raw_text = Path(file_bytes_or_path).read_text(encoding="utf-8-sig", errors="replace")
        else:
            try:
                raw_text = file_bytes_or_path.decode("utf-8-sig")
            except UnicodeDecodeError:
                raw_text = file_bytes_or_path.decode("cp1252", errors="replace")
        cleaned_txt = raw_text.strip()
        if len(cleaned_txt) < 10:
            raise ExtractionError("El archivo de texto no contiene contenido suficiente para indexar.")
        res = {
            "text": cleaned_txt,
            "format": ext.lstrip("."),
            "char_count": len(cleaned_txt)
        }
    else:
        raise ExtractionError(f"Formato no soportado: '{ext}'. Formatos admitidos: .pdf, .docx, .doc, .md, .txt")

    # Infer or clean document title from filename if no H1 exists
    title = Path(filename).stem.replace("-", " ").replace("_", " ").title()
    res["inferred_title"] = title
    res["filename"] = filename
    return res
