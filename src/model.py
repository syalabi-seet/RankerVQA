import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.models.deit.modeling_deit import DeiTModel
from transformers.models.beit.modeling_beit import BeitModel
from peft import get_peft_model, LoraConfig

class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        image_model_name="facebook/deit-small-distilled-patch16-224",
        proj_dim=128,
        hidden_dim=256,
        proj_depth=2,
        dropout=0.1,
        lora_r=8,
        lora_alpha=16,
    ):
        super().__init__()

        # ----- Text Encoder -----
        text_lora_cfg = LoraConfig(
            base_model_name_or_path=text_model_name,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            bias="none"
        )
        self.text_encoder = get_peft_model(AutoModel.from_pretrained(text_model_name), text_lora_cfg)
        self.text_proj = self.build_mlp(
            input_dim=self.text_encoder.config.hidden_size,
            output_dim=proj_dim,
            hidden_dim=hidden_dim,
            depth=proj_depth,
            dropout=dropout
        )

        # ----- Image Encoder -----
        image_cls = BeitModel if "beit" in image_model_name else DeiTModel
        image_lora_cfg = LoraConfig(
            base_model_name_or_path=image_model_name,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            bias="none"
        )
        self.image_encoder = get_peft_model(image_cls.from_pretrained(image_model_name), image_lora_cfg)
        self.image_proj = self.build_mlp(
            input_dim=self.image_encoder.config.hidden_size,
            output_dim=proj_dim,
            hidden_dim=hidden_dim,
            depth=proj_depth,
            dropout=dropout
        )

    def build_mlp(self, input_dim, output_dim, hidden_dim, depth, dropout):
        layers = []
        for i in range(depth):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(output_dim))  # Normalize final embedding
        return nn.Sequential(*layers)
    
    def get_image_embedding(self, image_inputs):
        """Returns normalized image embeddings from pixel input."""
        image_outputs = self.image_encoder(**image_inputs)
        cls_image = image_outputs.last_hidden_state[:, 0, :]
        image_embed = self.image_proj(cls_image)
        return F.normalize(image_embed, dim=-1)

    def get_text_embedding(self, text_inputs):
        """Returns normalized text embeddings from tokenized input."""
        text_outputs = self.text_encoder(**text_inputs)
        cls_text = text_outputs.last_hidden_state[:, 0, :]
        text_embed = self.text_proj(cls_text)
        return F.normalize(text_embed, dim=-1)
    
    def predict(self, image_tensor, tokenized_text_inputs, return_logits=False):
        """
        Args:
            image_tensor: torch.Tensor of shape [3, H, W] (already on correct device)
            tokenized_text_inputs: Dict of tokenized text tensors (already on correct device)
            return_logits: If True, return dot product scores instead of cosine similarity

        Returns:
            scores: torch.Tensor of shape [N], similarity scores between image and each caption
        """
        self.eval()

        with torch.no_grad():
            # Add batch dimension to image
            image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]

            # Get embeddings
            image_embed = self.get_image_embedding(image_tensor)           # [1, D]
            text_embeds = self.get_text_embedding(tokenized_text_inputs)  # [N, D]

            # Compute similarity
            if return_logits:
                scores = image_embed @ text_embeds.T  # [1, N]
            else:
                scores = F.cosine_similarity(image_embed, text_embeds)  # [N]

        return scores.squeeze(0)  # [N]