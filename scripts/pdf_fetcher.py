# pdf_fetcher.py

"""
Download core regulatory compliance PDFs for GDPR, HIPAA, and SOX.
This script uses browser-like headers and error handling to avoid 403 errors.
"""

import os
import requests

# Create a directory to store the downloaded PDFs
PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# Mapping of regulation IDs to their official PDF URLs
# These documents are in the public domain and freely accessible
REGULATIONS = {
    "GDPR": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
    "HIPAA": "https://www.hhs.gov/sites/default/files/hipaa-simplification-201303.pdf",
    "SOX": "https://www.govinfo.gov/content/pkg/COMPS-1883/pdf/COMPS-1883.pdf",  # GovInfo PDF with text layer
    "CCPA": "https://oag.ca.gov/sites/all/files/agweb/pdfs/privacy/ccpa-text-of-mod-clean-020720.pdf",  # California Consumer Privacy Act
    "PCI-DSS": "https://assets.kpmg.com/content/dam/kpmgsites/in/pdf/2024/08/payment-card-industry-data-security-standard-version-4.0.1.pdf",  # Payment Card Industry Data Security Standard
}

# Browser-like headers to mimic a real user and bypass simple bot filters
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf",  # We expect PDF responses
    "Accept-Language": "en-US,en;q=0.9",  # Preference for English
}


def download_pdfs(regs: dict, save_dir: str):
    """
    Download each regulation PDF using requests.
    - Uses custom headers to avoid 403 Forbidden errors from servers.
    - Follows redirects automatically.
    - Times out after 30 seconds to prevent hanging.
    - Handles HTTP and other exceptions gracefully.
    """
    for reg_id, url in regs.items():
        file_path = os.path.join(save_dir, f"{reg_id}.pdf")
        print(f"Downloading {reg_id} from {url}...")
        try:
            # Send GET request with headers and follow redirects
            response = requests.get(
                url, headers=HEADERS, allow_redirects=True, timeout=30
            )
            # Raise an HTTPError for bad status codes (e.g., 403, 404)
            response.raise_for_status()
            # Write binary PDF content to a file
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"  ✔ Saved to {file_path}")
        except requests.HTTPError as e:
            # Specific handling for HTTP errors with status code
            print(f"  ✖ HTTP error for {reg_id}: {e.response.status_code}")
        except Exception as e:
            # Generic handling for network, file I/O, or other errors
            print(f"  ✖ Failed to download {reg_id}: {e}")


if __name__ == "__main__":
    download_pdfs(REGULATIONS, PDF_DIR)
