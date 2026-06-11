import fitz  # PyMuPDF
import re
import json
import os

class CECProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.data = []

    def extract_hierarchy_and_text(self):
        """
        Iterates through the PDF and extracts text with Section, Rule, and Page metadata.
        """
        current_section = "Unknown Section"
        current_rule = "Unknown Rule"
        
        # Regex patterns for CEC structure
        section_pattern = re.compile(r"Section\s+(\d+)\s*—\s*(.*)", re.IGNORECASE)
        rule_pattern = re.compile(r"Rule\s+(\d+-\d+)\s*(.*)", re.IGNORECASE)

        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            text = page.get_text("text")
            lines = text.split('\n')
            
            page_text_blocks = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for Section header
                section_match = section_pattern.match(line)
                if section_match:
                    current_section = f"Section {section_match.group(1)}: {section_match.group(2)}"
                    continue
                
                # Check for Rule header
                rule_match = rule_pattern.match(line)
                if rule_match:
                    current_rule = f"Rule {rule_match.group(1)}: {rule_match.group(2)}"
                    continue
                
                page_text_blocks.append(line)

            # Join blocks into a single string for the page for now
            # In a more advanced version, we would chunk by Rule boundaries
            content = " ".join(page_text_blocks)
            
            if content.strip():
                self.data.append({
                    "text": content,
                    "metadata": {
                        "page": page_num + 1,
                        "section": current_section,
                        "rule": current_rule
                    }
                })

    def save_processed_data(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)
        print(f"Processed {len(self.data)} pages. Data saved to {output_path}")

if __name__ == "__main__":
    # Path relative to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file = os.path.join(base_dir, "..", "2024 CEC code book.pdf")
    output_file = os.path.join(base_dir, "cec_processed_data.json")
    
    if os.path.exists(pdf_file):
        processor = CECProcessor(pdf_file)
        processor.extract_hierarchy_and_text()
        processor.save_processed_data(output_file)
    else:
        print(f"Error: PDF not found at {pdf_file}")
