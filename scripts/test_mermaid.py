#!/usr/bin/env python3
"""
Test script for mermaid-py in the container
"""

import os
import sys

os.environ["MERMAID_INK_SERVER"] = "https://mermaid.ink"
print(f"Using MERMAID_INK_SERVER: {os.environ.get('MERMAID_INK_SERVER', 'not set')}")

print("Testing mermaid-py import and functionality...")

try:
    import mermaid as md

    print("✅ Successfully imported mermaid-py")
    print(
        f"Mermaid-py version: {md.__version__ if hasattr(md, '__version__') else 'unknown'}"
    )
except ImportError as e:
    print(f"❌ Failed to import mermaid-py: {e}")
    print("Try installing it with: pip install mermaid-py")
    sys.exit(1)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

mindmap_content = """mindmap
  Haskell for Imperative Programmers
    Prerequisites
      Imperative Programming
    Functional Programming
      Pure Functions
"""

video_id = "Vgu82wiiZ90"
mindmap_file = f"{output_dir}/{video_id}_mindmap.md"

if os.path.exists(mindmap_file):
    print(f"Found existing mindmap file: {mindmap_file}")
    try:
        with open(mindmap_file, "r") as f:
            file_content = f.read()
        print(f"Successfully read mindmap file, length: {len(file_content)} chars")

        import re

        if "```mermaid" in file_content:
            mermaid_match = re.search(r"```mermaid\s*(.*?)```", file_content, re.DOTALL)
            if mermaid_match:
                mindmap_content = mermaid_match.group(1).strip()
                print("Successfully extracted mermaid content from file")
            else:
                print(
                    "WARNING: Could not extract mermaid content, using sample content instead"
                )
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"No existing mindmap file found at {mindmap_file}, using sample content")

try:
    print("Creating Mermaid diagram object...")
    diagram = md.Mermaid(mindmap_content)

    print(
        "Available methods:",
        ", ".join([m for m in dir(diagram) if not m.startswith("_")]),
    )

    print("\nMermaid content being used:")
    print("---------------------------")
    print(mindmap_content)
    print("---------------------------")

    try:
        print("\nTrying to_svg method first...")
        svg_path = f"{output_dir}/test_mermaid_output.svg"
        diagram.to_svg(svg_path)

        if os.path.exists(svg_path):
            filesize = os.path.getsize(svg_path)
            print(
                f"✅ Successfully created SVG image: {svg_path} ({filesize/1024:.2f} KB)"
            )

            with open(svg_path, "r") as f:
                svg_content = f.read()
                print(f"SVG content length: {len(svg_content)} chars")
                print(f"SVG content starts with: {svg_content[:100]}...")
        else:
            print(f"❌ SVG file not created at {svg_path}")
    except Exception as svg_error:
        print(f"❌ Error generating SVG: {svg_error}")

    try:
        print("\nTrying to_png method...")
        output_path = f"{output_dir}/test_mermaid_output.png"

        print(f"Calling diagram.to_png({output_path})")
        diagram.to_png(output_path)

        if os.path.exists(output_path):
            filesize = os.path.getsize(output_path)
            print(
                f"✅ Successfully created PNG image: {output_path} ({filesize/1024:.2f} KB)"
            )

            if filesize == 0:
                print("⚠️ WARNING: PNG file has zero size!")
        else:
            print(f"❌ PNG file not created at {output_path}")
    except Exception as e:
        print(f"❌ Error generating PNG: {e}")
        import traceback

        traceback.print_exc()

    try:
        print("\nCreating HTML test file to view SVG...")
        svg_path = f"{output_dir}/test_mermaid_output.svg"
        html_path = f"{output_dir}/test_mermaid_viewer.html"

        if os.path.exists(svg_path):
            with open(svg_path, "r") as f:
                svg_content = f.read()

            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mermaid SVG Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .svg-container {{ 
            border: 1px solid #ddd; 
            padding: 10px;
            border-radius: 5px;
            background-color: #f9f9f9;
            margin: 20px 0;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        button {{
            padding: 5px 10px;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <h1>Mermaid SVG Viewer</h1>
    <div class="controls">
        <button onclick="zoomIn()">Zoom In</button>
        <button onclick="zoomOut()">Zoom Out</button>
        <button onclick="resetZoom()">Reset</button>
    </div>
    <div class="svg-container" id="container">
        {svg_content}
    </div>
    
    <script>
        // Get the SVG element
        const svg = document.querySelector('svg');
        let scale = 1;
        
        // Add zoom functionality
        function zoomIn() {{
            scale += 0.1;
            svg.style.transform = `scale(${{scale}})`;
            svg.style.transformOrigin = 'top left';
        }}
        
        function zoomOut() {{
            if (scale > 0.2) {{
                scale -= 0.1;
                svg.style.transform = `scale(${{scale}})`;
                svg.style.transformOrigin = 'top left';
            }}
        }}
        
        function resetZoom() {{
            scale = 1;
            svg.style.transform = `scale(${{scale}})`;
        }}
    </script>
</body>
</html>
"""

            with open(html_path, "w") as f:
                f.write(html_content)

            print(f"✅ Created HTML viewer at {html_path}")
        else:
            print(f"❌ Cannot create HTML viewer, SVG file not found at {svg_path}")
    except Exception as e:
        print(f"❌ Error creating HTML viewer: {e}")

except Exception as e:
    print(f"❌ Mermaid diagram creation failed: {e}")
    import traceback

    traceback.print_exc()

print("\nTest completed!")
