import os
from pathlib import Path
os.environ["XDG_CACHE_HOME"] = str(Path("/data") / "ahmed" / ".cache")

import re

from argparse import ArgumentParser

import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from tqdm.auto import tqdm
from loguru import logger

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.renderers.json import JSONRenderer

def extract_page_order(node_id):
    """
    Extracts page number and node order from the node ID.

    Args:
        node_id (str): Node identifier in format "/page/0/Text/9".

    Returns:
        (int, int): (page_number, node_order) extracted from the node_id.
    """
    match = re.search(r'/page/(\d+)/(\w+)/(\d+)', node_id)
    if match:
        return int(match.group(1)), str(match.group(2)), int(match.group(3))
    return None, None

class GraphBuilder:
    def __init__(self, device):
        self.device = device

        self.assets_dir = Path("assets")
        self.pdf_converter = PdfConverter(artifact_dict=create_model_dict(device=self.device))
        self.json_renderer = self.pdf_converter.resolve_dependencies(JSONRenderer)

    def _get_output(self, in_path: Path):
        document = self.pdf_converter.build_document(str(in_path))
        json_output = self.json_renderer(document)
        return json_output, document.pages
    
    def build_nodes(self, json_output, parent_id=None):
        """
        Recursively processes nodes, ensuring that parent nodes contain 
        their own HTML as well as a list of the HTML content of their children.

        Args:
            json_output: JSON object representing the node.
            parent_id: ID of the parent node.

        Returns:
            A list of processed nodes with parent nodes storing child HTML as a list.
        """
        children = getattr(json_output, "children", [])
        html_content = getattr(json_output, "html", "").strip()
        node_id = getattr(json_output, "id", None)
        block_type = getattr(json_output, "block_type", None)

        bbox = getattr(json_output, "bbox", None)
        images = getattr(json_output, "images", None)
        if node_id:
            page_no, _, order_no = extract_page_order(node_id)
        else:
            page_no = -1
            order_no = -1

        child_nodes = []  # Store processed child nodes

        if children:
            # Process children recursively
            for child in children:
                processed_children = self.build_nodes(child, node_id)
                child_nodes.extend(processed_children)  # Add child nodes to the list

        soup = BeautifulSoup(html_content, "html.parser")
        text_content = soup.get_text(strip=True)

        # Store the node with child HTML as a list
        node_data = {
            "node_id": node_id,
            "text": text_content,
            "parent_id": parent_id,
            "block_type": block_type,
            "page_no": page_no,
            "order_no": order_no,
            "bbox": [int(i) for i in bbox] if bbox else None,
            "images": images,
            "children": [n["node_id"] for n in child_nodes]
        }

        return [node_data] + child_nodes  # Return parent + children
    
    def get_image_captions_heuristic(self, elements):
        visual_types = ["Figure", "Table"]
        caption_like_types = ["Caption", "Text", "TextInlineMath"]

        visual_elements = elements[elements["block_type"].isin(visual_types)]
        caption_candidates = elements[elements["block_type"].isin(caption_like_types)]

        compiled = []

        for _, visual in visual_elements.iterrows():
            v_page = visual["page_no"]
            v_bbox = visual["bbox"]  # [x1, y1, x2, y2]
            v_x1, v_y1, v_x2, v_y2 = v_bbox

            same_page_text = caption_candidates[caption_candidates["page_no"] == v_page]

            best_above = None
            best_below = None
            min_dist_above = float("inf")
            min_dist_below = float("inf")

            for _, candidate in same_page_text.iterrows():
                c_bbox = candidate["bbox"]
                c_x1, c_y1, c_x2, c_y2 = c_bbox

                horizontal_overlap = max(0, min(v_x2, c_x2) - max(v_x1, c_x1))
                overlap_ratio = horizontal_overlap / (v_x2 - v_x1)

                if overlap_ratio < 0.3:
                    continue  # Not aligned enough

                # Caption is above
                if c_y2 <= v_y1:
                    distance = v_y1 - c_y2
                    if distance < min_dist_above:
                        min_dist_above = distance
                        best_above = candidate

                # Caption is below
                elif c_y1 >= v_y2:
                    distance = c_y1 - v_y2
                    if distance < min_dist_below:
                        min_dist_below = distance
                        best_below = candidate

            for position, caption in [("above", best_above), ("below", best_below)]:
                if caption is not None:
                    compiled.append({
                        "document_id": visual["document_id"],
                        "block_type": visual["block_type"],
                        "node_id": visual["node_id"],
                        "page_no": visual["page_no"],
                        "order_no": visual["order_no"],
                        "bbox": visual["bbox"],
                        "caption": caption["text"],
                        "caption_position": position,
                        "distance_score": min_dist_above if position == "above" else min_dist_below,
                        "source": "heuristic-nearest"
                    })

        return compiled
    
    def resize_page(self, page):
        # Get low-resolution PIL image
        page_image = page.lowres_image

        # Get mediabox size from the page geometry (likely in points or pixels)
        mediabox_size = tuple([int(i) for i in page.polygon.bbox[-2:]])  # width, height

        # Resize the image to match mediabox
        resized_image = page_image.resize(mediabox_size, resample=Image.BILINEAR)

        return resized_image
    
    def build_data(self, path):
        visual_types = [
            "Figure", "FigureGroup",
            "Table", "TableGroup"
        ]

        json_output, page_images = builder._get_output(path)
        page_images = [self.resize_page(page) for page in page_images]
        elements = pd.DataFrame(self.build_nodes(json_output))
        elements["document_id"] = Path(path).stem
        caption_elements = pd.DataFrame(self.get_image_captions_heuristic(elements))

        # Drop the visual elements out of the remaining elements
        elements = pd.DataFrame(elements[~elements["block_type"].isin(visual_types)].reset_index(drop=True))
        elements = pd.DataFrame(elements[elements["text"].isna() | elements["text"].ne("")]).reset_index(drop=True)
        elements = elements[["document_id", "block_type", "node_id", "page_no", "bbox", "text"]]
        return elements, caption_elements, page_images
    

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Reverse the order or behavior when set'
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()

    builder = GraphBuilder(args.device)

    asset_dir = Path("assets")
    in_dir = asset_dir / "pdf10000"
    in_paths = sorted(list(in_dir.glob("*.pdf")))

    if args.reverse:
        in_paths = in_paths[::-1]

    for in_path in tqdm(in_paths):
        doc_id = in_path.stem
        out_dir = asset_dir / "processed" / f"{doc_id}"

        if out_dir.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            elements, captions, page_images = builder.build_data(in_path)

            # Save elements
            elements.to_parquet(out_dir / "elements.parquet")
            
            # Save captions
            captions.to_parquet(out_dir / "captions.parquet")

            # Save page images
            for i, page_image in enumerate(page_images):
                page_dir = out_dir / "images"
                page_dir.mkdir(parents=True, exist_ok=True)
                page_image.save(page_dir / f"page_{i}.png")
        except Exception as e:
            logger.error(f"{doc_id}: {e}")