
from models.dit_text import DiTTextPipeLine
from diffusers import (DiTPipeline,
                StableDiffusionPipeline, 
                AutoencoderKL,
                Transformer2DModel,
                )
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTokenizer,CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


def test_encode(pipeline: DiTTextPipeLine,
                prompt: str,
                num_images_per_prompt: int = 1,
                do_classifier_guidance: bool = False,
                negative_prompt = None,
                ):
    try:
        cond, uncond = pipeline.encode_prompt(device='cpu', 
                               prompt=prompt,
                               num_images_per_prompt=num_images_per_prompt,
                               do_classifier_free_guidance=do_classifier_guidance,
                               negative_prompt=negative_prompt)
        out_dict = {"output": (cond, uncond), "pass": True}
        return out_dict
    
    except Exception as e:
        out_dict = {"output": e, "pass": False}
        return out_dict
    

if __name__ == "__main__":
    transformer_cfg_dict = {
        "_class_name": "Transformer2DModel",
        "_diffusers_version": "0.12.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 72,
        "cross_attention_dim": None,
        "dropout": 0.0,
        "in_channels": 4,
        "norm_elementwise_affine": False,
        "norm_num_groups": 32,
        "norm_type": "ada_norm_zero",
        "num_attention_heads": 16,
        "num_embeds_ada_norm": 1000,
        "num_layers": 28,
        "num_vector_embeds": None,
        "only_cross_attention": False,
        "out_channels": 8,
        "patch_size": 2,
        "sample_size": 64,
        "upcast_attention": False,
        "use_linear_projection": False
    }
    
    model = DiTTextPipeLine(
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae"),
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder"),
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer"),
        transformer = Transformer2DModel.from_config(transformer_cfg_dict),
        scheduler = KarrasDiffusionSchedulers(1),
        feature_extractor= CLIPImageProcessor(),
    )
    prompt = "a cat"
    num_images_per_prompt = 1
    do_classifier_guidance = True
    negative_prompt = None
    res = test_encode(model, prompt, num_images_per_prompt, do_classifier_guidance, negative_prompt)
    print(res['output'][0].shape, res['output'], res['pass'])
