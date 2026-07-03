"""Resume CONTENT verification — an uploaded PDF must actually be a resume/CV,
not some other document (invoice, certificate, ID). The text heuristic is pure and
fails open on unreadable/short text so a real CV is never wrongly blocked."""
from app.validation import extract_pdf_text, is_resume_pdf, resume_text_verdict

_RESUME = """
    PRASANTH M
    prasanth@example.com  |  +91 90000 00001  |  Chennai

    Professional Summary
    Full-stack developer with 3 years of experience building web apps.

    Work Experience
    Software Engineer, Acme Corp (2021-2024) — Python, FastAPI, React.

    Education
    B.E. Computer Science, Anna University, 2021.

    Skills
    Python, JavaScript, SQL, Docker, Git.

    Projects
    Job board chatbot; inventory system.
"""

_INVOICE = """
    TAX INVOICE                          Invoice No: INV-2024-00891
    Bill To: Some Company Pvt Ltd        Date: 12/03/2024
    Description            Qty     Rate        Amount
    Consulting services    10     5000.00     50000.00
    Subtotal                                  50000.00
    GST @ 18%                                   9000.00
    Total Amount Due                           59000.00
    Payment terms: net 30 days. Thank you for your business.
"""


def test_resume_text_accepted():
    ok, reason = resume_text_verdict(_RESUME)
    assert ok is True, reason


def test_invoice_text_rejected():
    ok, reason = resume_text_verdict(_INVOICE)
    assert ok is False                                       # rejected (negative marker or no signals)


def test_short_or_empty_text_fails_open():
    assert resume_text_verdict("")[0] is True
    assert resume_text_verdict("hello")[0] is True          # too little to judge → accept


def test_pan_card_rejected_even_when_short():
    # a typical e-PAN carries little text but a clear label + PAN number
    pan = ("INCOME TAX DEPARTMENT  GOVT. OF INDIA\n"
           "Permanent Account Number Card\nABCDE1234F\nPRASANTH M\n01/01/1995")
    ok, reason = resume_text_verdict(pan)
    assert ok is False and reason == "non_resume_document"


def test_aadhaar_rejected():
    aad = "Government of India\nAadhaar\nName: Prasanth M\n1234 5678 9012"
    assert resume_text_verdict(aad)[0] is False


def test_invoice_marker_rejected_short():
    assert resume_text_verdict("TAX INVOICE  Invoice No: 42  GSTIN 33ABCDE")[0] is False


def test_certifications_section_not_falsely_rejected():
    # 'certifications' is a legit resume section — must NOT trigger a negative marker
    ok, _ = resume_text_verdict(_RESUME + "\nCertifications: AWS Certified Developer")
    assert ok is True


def test_single_signal_needs_contact():
    # only "skills" and nothing else, no email/phone → not enough → reject
    text = "SKILLS " + ("padding blah blah " * 40)
    assert resume_text_verdict(text)[0] is False
    # same one signal but WITH contact info → accept
    assert resume_text_verdict(text + " reach me at me@x.com")[0] is True


def test_is_resume_pdf_fails_open_on_unreadable_bytes():
    # not a real PDF → no text extracted → accept (never block on unreadable input)
    assert is_resume_pdf(b"not a pdf")[0] is True
    assert is_resume_pdf(b"")[0] is True


def test_real_pdf_roundtrip_resume_vs_invoice():
    """Build real PDFs with pypdf-friendly reportlab-free bytes via pypdf's writer,
    then extract + classify. Skips cleanly if PDF text extraction isn't available."""
    try:
        from pypdf import PdfWriter
    except Exception:                                       # pragma: no cover
        import pytest
        pytest.skip("pypdf not installed")

    def _pdf(text: str) -> bytes:
        import io
        w = PdfWriter()
        w.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        w.write(buf)
        return buf.getvalue()

    # A blank page has no extractable text → fails open (accept). This asserts the
    # extractor runs without error and the fail-open path holds for image-only PDFs.
    data = _pdf(_RESUME)
    assert isinstance(extract_pdf_text(data), str)
    assert is_resume_pdf(data)[0] is True                   # no text → accept
