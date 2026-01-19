import json
import base64
import os
from pathlib import Path

# Load the notebook
notebook_path = 'notebooks/comprehensive_experiments.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create output directories
output_dir = Path('extracted_results')
images_dir = output_dir / 'images'
text_results_dir = output_dir / 'text_results'

images_dir.mkdir(parents=True, exist_ok=True)
text_results_dir.mkdir(parents=True, exist_ok=True)

# Track statistics
image_count = 0
text_output_count = 0
table_count = 0

print(f"📓 Analyzing notebook: {notebook_path}")
print(f"Total cells: {len(nb['cells'])}")

# Process each cell
for cell_idx, cell in enumerate(nb['cells']):
    outputs = cell.get('outputs', [])
    
    for output_idx, output in enumerate(outputs):
        # Extract PNG images
        if 'data' in output and 'image/png' in output['data']:
            image_count += 1
            image_data = output['data']['image/png']
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Save image
            image_filename = f"cell_{cell_idx:03d}_output_{output_idx:02d}_image_{image_count:03d}.png"
            image_path = images_dir / image_filename
            with open(image_path, 'wb') as img_file:
                img_file.write(image_bytes)
            
            print(f"  ✓ Extracted image: {image_filename}")
        
        # Extract text outputs (results, metrics, etc.)
        if 'text' in output:
            text_output_count += 1
            text_content = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
            
            # Save text output
            text_filename = f"cell_{cell_idx:03d}_output_{output_idx:02d}_text_{text_output_count:03d}.txt"
            text_path = text_results_dir / text_filename
            with open(text_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text_content)
            
            # Print preview of important results
            if any(keyword in text_content.lower() for keyword in ['f1', 'precision', 'recall', 'accuracy', 'results', 'experiment']):
                preview = text_content[:200].replace('\n', ' ')
                print(f"  ℹ️  Found results text: {preview}...")
        
        # Extract HTML tables (if any)
        if 'data' in output and 'text/html' in output['data']:
            table_count += 1
            html_content = ''.join(output['data']['text/html']) if isinstance(output['data']['text/html'], list) else output['data']['text/html']
            
            # Save HTML table
            html_filename = f"cell_{cell_idx:03d}_output_{output_idx:02d}_table_{table_count:03d}.html"
            html_path = text_results_dir / html_filename
            with open(html_path, 'w', encoding='utf-8') as html_file:
                html_file.write(html_content)
            
            print(f"  ✓ Extracted table: {html_filename}")

# Generate summary report
summary_path = output_dir / 'extraction_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as summary:
    summary.write("="*60 + "\n")
    summary.write("COMPREHENSIVE EXPERIMENTS - EXTRACTION SUMMARY\n")
    summary.write("="*60 + "\n\n")
    summary.write(f"Source Notebook: {notebook_path}\n")
    summary.write(f"Total Cells: {len(nb['cells'])}\n\n")
    summary.write(f"Extracted Results:\n")
    summary.write(f"  - Images (PNG): {image_count}\n")
    summary.write(f"  - Text Outputs: {text_output_count}\n")
    summary.write(f"  - HTML Tables: {table_count}\n\n")
    summary.write(f"Output Directories:\n")
    summary.write(f"  - Images: {images_dir}\n")
    summary.write(f"  - Text Results: {text_results_dir}\n")
    summary.write("="*60 + "\n")

print("\n" + "="*60)
print(f"✅ Extraction Complete!")
print(f"   - {image_count} images saved to: {images_dir}")
print(f"   - {text_output_count} text outputs saved to: {text_results_dir}")
print(f"   - {table_count} HTML tables saved to: {text_results_dir}")
print(f"   - Summary saved to: {summary_path}")
print("="*60)

# List all extracted files
print("\n📁 Extracted Files:\n")
print("Images:")
for img in sorted(images_dir.glob('*.png')):
    size_kb = img.stat().st_size / 1024
    print(f"  - {img.name} ({size_kb:.1f} KB)")

if text_output_count > 0:
    print("\nText Results:")
    for txt in sorted(text_results_dir.glob('*.txt'))[:10]:  # Show first 10
        print(f"  - {txt.name}")
    if text_output_count > 10:
        print(f"  ... and {text_output_count - 10} more")

if table_count > 0:
    print("\nHTML Tables:")
    for html in sorted(text_results_dir.glob('*.html')):
        print(f"  - {html.name}")
