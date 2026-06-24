# Modified from https://github.com/DeepSeek-AI/Janus/blob/main/janus/models/modeling_vlm.py

import os
from pathlib import Path

import numpy as np
import torch
import transformers.modeling_outputs
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower

from . import processing_vlm as _processing_vlm
from . import utils as _utils
from ._attrdict_patched import AttrDict
from .projector import MlpProjector

# FIXME: can we calculate those values somehow?
IMAGE_TOKEN_NUM_PER_IMAGE = 576
PATCH_SIZE = 16


def _resolve_model_path(model_path: str | os.PathLike) -> str | Path:
    path = Path(model_path)
    if path.exists():
        return path
    local_janus = Path(__file__).resolve().parents[2] / "model" / "Janus-Pro-7B"
    if str(model_path) == "deepseek-ai/Janus-Pro-7B" and local_janus.exists():
        return local_janus
    return str(model_path)


def load_model(
    model_path: str | os.PathLike,
) -> tuple["MultiModalityCausalLM", _processing_vlm.VLChatProcessor]:
    resolved_model_path = _resolve_model_path(model_path)
    processor_path = resolved_model_path if Path(resolved_model_path).exists() else "deepseek-ai/Janus-Pro-7B"
    processor = _processing_vlm.VLChatProcessor.from_pretrained(processor_path, use_fast=True)

    # FIXME: What's the dtype? Config looks like bf16
    model = MultiModalityCausalLM.from_pretrained(
        resolved_model_path,
        trust_remote_code=False,
    )

    assert all(
        token in processor.tokenizer.vocab
        for token in (
            processor.image_start_tag,
            processor.image_end_tag,
            processor.pad_tag,
        )
    )

    return model, processor


class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(params.n_embed, params.image_token_embed)
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(params.image_token_embed, params.image_token_size)

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    # Label smoothing for training text to image
    # FIXME: is this the best place to add this?
    label_smoothing: float = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed)

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    @torch.inference_mode()
    def generate_images(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        cfg_weight: float,
        temperature: float,
        image_token_num_per_image: int = IMAGE_TOKEN_NUM_PER_IMAGE,
        img_size: int = _utils.IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
    ) -> np.ndarray:
        true_batch_size = input_ids.shape[0]  # batch size before CFG duplication
        generated_tokens = torch.zeros(
            (true_batch_size, image_token_num_per_image),
            device=input_ids.device,
            dtype=torch.int,
        )

        if cfg_weight != 1.0:
            # Duplicate input_ids for CFG
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

            # CFG tokens are second half; they should be BOS, all padding, then SOI
            input_ids[input_ids.shape[0] // 2 :, 0] = 100000  # FIXME: BOS; un-hardcode
            input_ids[input_ids.shape[0] // 2 :, -1] = 100016  # FIXME: SOI; un-hardcode
            input_ids[input_ids.shape[0] // 2 :, 1:-1] = 100015  # FIXME: Padding; un-hardcode
            attention_mask[input_ids.shape[0] // 2 :, :] = True  # attention for all unconditional tokens

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]  # batch size after CFG duplication if applicable
        embedding_dim = inputs_embeds.shape[-1]

        outputs = None
        for img_token_idx in range(image_token_num_per_image):
            outputs = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=outputs.past_key_values if img_token_idx != 0 else None,
            )
            hidden_states = outputs.last_hidden_state

            logits = self.gen_head(hidden_states[:, -1, :])

            # Apply CFG if specified
            if cfg_weight != 1.0:
                # First half is conditional, second half is unconditional
                logit_cond = logits[: input_ids.shape[0] // 2, :]
                logit_uncond = logits[input_ids.shape[0] // 2 :, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            # Sample next token
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)[:, 0]
            else:
                next_token = torch.argmax(logits, dim=-1)
            assert next_token.shape == (true_batch_size,)
            generated_tokens[:, img_token_idx] = next_token

            if cfg_weight != 1.0:
                # Use generated tokens for both conditional and unconditional
                next_token = next_token.repeat(2)
            assert next_token.shape == (batch_size,)
            img_embeds = self.prepare_gen_img_embeds(next_token)
            assert img_embeds.shape == (batch_size, embedding_dim)

            # New inputs are just the new embedding and the expanded full attention mask
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            assert inputs_embeds.shape == (batch_size, 1, embedding_dim)
            assert attention_mask.shape == (batch_size, input_ids.shape[1] + img_token_idx + 1)
            # FIXME: HF example here also uses cache position and shifts it: https://huggingface.co/docs/transformers/en/cache_explanation

        decoded_pixels = self.gen_vision_model.decode_code(
            generated_tokens,
            shape=(true_batch_size, 8, img_size // patch_size, img_size // patch_size),
        )
        decoded_pixels = decoded_pixels.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        decoded_pixels = np.clip((decoded_pixels + 1) / 2 * 255, 0, 255)
        assert decoded_pixels.shape == (true_batch_size, img_size, img_size, 3)

        return decoded_pixels.astype(np.uint8)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> transformers.modeling_outputs.CausalLMOutput:
        loss_type = kwargs.pop("loss_type")
        if loss_type == "text2image":
            return self.loss_image_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_ids=kwargs.pop("image_ids"),
                labels=labels,
                images_seq_mask=kwargs.pop("images_seq_mask"),
                **kwargs,
            )
        else:
            assert loss_type in ("image2text", "text2text")
            if loss_type == "image2text":
                raise NotImplementedError("Image inputs not implemented yet.")
            return self.loss_text_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

    def loss_image_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        image_ids: torch.LongTensor,
        labels: torch.LongTensor,
        images_seq_mask: torch.BoolTensor,
        **kwargs,
    ) -> transformers.modeling_outputs.CausalLMOutput:
        # Construct inputs_embeds for text and images
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Replace image tokens with their embeddings
        embedding_dim = input_embeds.shape[-1]
        image_embeds = self.prepare_gen_img_embeds(image_ids)
        input_embeds[images_seq_mask] = image_embeds.reshape(-1, embedding_dim)

        # Forward pass through the model
        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # Image generation uses the gen head instead of the lm head
        logits = self.gen_head(outputs.last_hidden_state)

        # Calculate loss
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(
            shift_logits,
            shift_labels,
            reduction="mean",
            label_smoothing=self.config.label_smoothing,
        )

        return transformers.modeling_outputs.CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def loss_text_generation(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        # Can just use the LlamaForCausalLM model directly here
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
