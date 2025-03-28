import random

from PIL import Image, ImageOps

import pandas as pd

from pathlib import Path

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, 
    DeiTImageProcessor,
)

class ImageCaptionDataset(Dataset):
    def __init__(
            self,
            split,
            n_samples=None,
            k = 10,
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            image_model_name="facebook/deit-small-distilled-patch16-224",
            device="cuda:3"
        ):
        """
        visual_df: DataFrame with visual elements (anchors), including 'caption'
        textual_df: DataFrame with textual elements (pairs), including 'caption'
        tokenizer: Optional tokenizer for caption or text (e.g. for model input)
        """
        self.split = split
        self.n_samples = n_samples
        self.k = k
        self.device = device
        self.processed_dir = Path("assets") / "processed"
        self.visual_df, self.qa_df = self.get_content()

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.extractor = DeiTImageProcessor.from_pretrained(image_model_name, use_fast=False)

    def get_content(self):
        qa_path = Path("assets") / "qa.parquet"
        if not qa_path.exists():
            train_qa_df = pd.read_csv("assets/mmvqa_qa_pairs_train_github.csv")
            train_qa_df["split"] = "train"
            val_qa_df = pd.read_csv("assets/mmvqa_qa_pairs_val_github.csv")
            val_qa_df["split"] = "val"
            qa_df = pd.concat([train_qa_df, val_qa_df])
            qa_df.to_parquet(qa_path)
        
        qa_df = pd.read_parquet(qa_path)
        qa_df = qa_df[qa_df["split"].eq(self.split)].reset_index(drop=True)

        caption_path = Path("assets") / "all_captions.parquet"
        if not caption_path.exists():
            caption_df = pd.concat([pd.read_parquet(d) for d in self.processed_dir.glob("*/captions.parquet")])
            caption_df["split"] = caption_df["document_id"].apply(lambda x: "train" if x in qa_df["document_id"].tolist() else "val")
            caption_df = caption_df[caption_df["caption"].ne("")].reset_index(drop=True)
            caption_df.to_parquet(caption_path)

        caption_df = pd.read_parquet(caption_path)
        caption_df = caption_df[caption_df["split"].eq(self.split)].reset_index(drop=True)

        if self.n_samples:
            caption_df = caption_df.sample(self.n_samples, random_state=42)
        return  caption_df, qa_df
    
    def encode_image(self, image):
        inputs = self.extractor(image, return_tensors="pt").to(self.device)
        return inputs
    
    def encode_text(self, texts):
        return self.tokenizer(
            [t if t!=None else "" for t in texts], 
            padding="max_length", 
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
    
    def get_cropped_image(self, anchor_row):
        page_image_path = self.processed_dir / f"{anchor_row['document_id']}" 
        page_image_path = page_image_path / "images" / f"page_{anchor_row['page_no']}.png"
        
        page_image = Image.open(page_image_path).convert("RGB")        
        cropped_image = page_image.crop(anchor_row["bbox"])  # crop the anchor

        # Step 1: Resize so the longer side = 224
        w, h = cropped_image.size
        scale = 224 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cropped_image.resize((new_w, new_h), Image.BILINEAR)

        # Step 2: Pad to 224×224
        delta_w = 224 - new_w
        delta_h = 224 - new_h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2)
        )
        padded = ImageOps.expand(resized, padding, fill=(255, 255, 255))  # white padding

        return padded

    def __len__(self):
        return len(self.visual_df)

    def __getitem__(self, idx):
        anchor_row = self.visual_df.iloc[idx].to_dict()
        caption = anchor_row['caption']

        # Tokenize the ground-truth caption (positive)
        if isinstance(caption, list) and len(caption) > 1:
            caption = random.choice(caption)
        caption_inputs = self.encode_text([caption])  # [1, seq_len]

        # Process the cropped image
        pix = self.get_cropped_image(anchor_row)
        pix_inputs = self.encode_image(pix)

        # Load the corresponding elements.parquet for this document
        doc_id = anchor_row["document_id"]
        element_path = self.processed_dir / doc_id / "elements.parquet"
        textual_df = pd.read_parquet(element_path)

        # Build caption lookup locally (optional, if you still need it later)
        caption_to_text_rows = {}
        for _, row in textual_df.iterrows():
            text = row["text"]
            if text not in caption_to_text_rows:
                caption_to_text_rows[text] = []
            caption_to_text_rows[text].append(row)

        same_page_df = textual_df[(textual_df['text'] != caption)]

        negative_candidates = same_page_df

        if len(negative_candidates) == 0:
            raise ValueError(f"No valid negative candidates found for doc {doc_id}")

        # Sample K negatives
        negative_samples = negative_candidates.sample(n=min(self.k, len(negative_candidates)))
        text_inputs = self.encode_text(negative_samples["text"].tolist())  # [K, seq_len]

        return pix_inputs, text_inputs, caption_inputs