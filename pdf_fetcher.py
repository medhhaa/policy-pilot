"""
Download core regulatory compliance PDFs for GDPR, HIPAA, and SOX.
Place this script at the root of your project.
"""

import os
import requests

# Create a directory to store the PDFs
PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# Mapping of regulation IDs to their official PDF URLs
REGULATIONS = {
    "GDPR": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
    "HIPAA": "https://www.hhs.gov/sites/default/files/hipaa-simplification-201303.pdf",
    "SOX": "https://www.sec.gov/about/laws/soa2002.pdf"
}

def download_pdfs(regs: dict, save_dir: str):
    for reg_id, url in regs.items():
        file_path = os.path.join(save_dir, f"{reg_id}.pdf")
        print(f"Downloading {reg_id} from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"  ✔ Saved to {file_path}")
        else:
            print(f"  ✖ Failed to download {reg_id} (Status {response.status_code})")

if __name__ == "__main__":
    download_pdfs(REGULATIONS, PDF_DIR)

