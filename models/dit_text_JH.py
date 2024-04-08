# from diffusers import DiTPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline
# import torch

# from typing import Dict, List, Optional, Tuple, Union

# import torch

# from ...models import AutoencoderKL, Transformer2DModel
# from ...schedulers import KarrasDiffusionSchedulers
# from ...utils.torch_utils import randn_tensor
# from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# # ======================================
# from typing import Any, Callable, Dict, List, Optional, Union

# import torch
# from packaging import version
# from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# from ...configuration_utils import FrozenDict
# from ...image_processor import PipelineImageInput, VaeImageProcessor
# from ...loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
# from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
# from ...models.lora import adjust_lora_scale_text_encoder
# from ...schedulers import KarrasDiffusionSchedulers
# from ...utils import (
#     USE_PEFT_BACKEND,
#     deprecate,
#     logging,
#     replace_example_docstring,
#     scale_lora_layers,
#     unscale_lora_layers,
# )
# from ...utils.torch_utils import randn_tensor
# from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
# from .pipeline_output import StableDiffusionPipelineOutput
# from .safety_checker import StableDiffusionSafetyChecker

from diffusers import (DiTPipeline,
                StableDiffusionPipeline, 
                AutoencoderKL,
                Transformer2DModel,
                )
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTokenizer,CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from collections import OrderedDict
from typing import Optional, Union, List, Tuple
from diffusers.utils.torch_utils import randn_tensor

import torch


class DiTTextPipeline(
    DiTPipeline
    # DiffusionPipeline,    
    # StableDiffusionMixin,
    # TextualInversionLoaderMixin,
    # LoraLoaderMixin,
    # IPAdapterMixin,
    # FromSingleFileMixin
    ):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`Transformer2DModel`]):
            A class conditioned `Transformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """
    # def __init__(self, *args):
    #     pass

    # model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        # unet: UNet2DConditionModel,
        transformer: Transformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
        # safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        # requires_safety_checker: bool = True,    

    # def __init__(
    #     self,
    #     transformer: Transformer2DModel,
    #     vae: AutoencoderKL,
    #     scheduler: KarrasDiffusionSchedulers,
    #     # id2label: Optional[Dict[int, str]] = None,
        ):
        super().__init__(
            transformer= transformer,
            vae= vae,
            scheduler= scheduler,
        )
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        # self.labels = {}
        # if id2label is not None:
        #     for key, value in id2label.items():
        #         for label in value.split(","):
        #             self.labels[label.lstrip().rstrip()] = int(key)
        #     self.labels = dict(sorted(self.labels.items()))

    # def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
    #     r"""

    #     Map label strings from ImageNet to corresponding class ids.

    #     Parameters:
    #         label (`str` or `dict` of `str`):
    #             Label strings to be mapped to class ids.

    #     Returns:
    #         `list` of `int`:
    #             Class ids to be processed by pipeline.
    #     """

    #     if not isinstance(label, list):
    #         label = list(label)

    #     for l in label:
    #         if l not in self.labels:
    #             raise ValueError(
    #                 f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
    #             )

    #     return [self.labels[l] for l in label]
    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
        ):
            r"""
            Encodes the prompt into text encoder hidden states.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                device: (`torch.device`):
                    torch device
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                do_classifier_free_guidance (`bool`):
                    whether to use classifier free guidance or not
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                lora_scale (`float`, *optional*):
                    A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
            """
            # set lora scale so that monkey patched LoRA
            # function of text encoder can correctly access it
            # if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            #     self._lora_scale = lora_scale

            #     # dynamically adjust the LoRA scale
            #     if not USE_PEFT_BACKEND:
            #         adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            #     else:
            #         scale_lora_layers(self.text_encoder, lora_scale)

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:
                # textual inversion: process multi-vector tokens if necessary
                # if isinstance(self, TextualInversionLoaderMixin):
                #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    # logger.warning(
                    #     "The following part of your input was truncated because CLIP can only handle sequences up to"
                    #     f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    # )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                if clip_skip is None:
                    prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                    prompt_embeds = prompt_embeds[0]
                else:
                    prompt_embeds = self.text_encoder(
                        text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                    )
                    # Access the `hidden_states` first, that contains a tuple of
                    # all the hidden states from the encoder layers. Then index into
                    # the tuple to access the hidden states from the desired layer.
                    prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                    # We also need to apply the final LayerNorm here to not mess with the
                    # representations. The `last_hidden_states` that we typically use for
                    # obtaining the final prompt representations passes through the LayerNorm
                    # layer.
                    prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                # textual inversion: process multi-vector tokens if necessary
                # if isinstance(self, TextualInversionLoaderMixin):
                #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = uncond_input.attention_mask.to(device)
                else:
                    attention_mask = None

                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            #     # Retrieve the original scale by scaling back the LoRA layers
            #     unscale_lora_layers(self.text_encoder, lora_scale)

            return prompt_embeds, negative_prompt_embeds

    # @torch.no_grad()
    def __call__(
        self,
        # class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        #=====================
        # self,
        prompt: Union[str, List[str]] = None,
        # height: Optional[int] = None,
        # width: Optional[int] = None,
        # num_inference_steps: int = 50,
        # timesteps: List[int] = None,
        # guidance_scale: float = 7.5,
        # negative_prompt: Optional[Union[str, List[str]]] = None,
        # num_images_per_prompt: Optional[int] = 1,
        # eta: float = 0.0,
        # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # latents: Optional[torch.FloatTensor] = None,
        # prompt_embeds: Optional[torch.FloatTensor] = None,
        # negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        # ip_adapter_image: Optional[PipelineImageInput] = None,
        # ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        # output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # guidance_rescale: float = 0.0,
        # clip_skip: Optional[int] = None,
        # callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        # class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        # class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        # class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
        # cross attention K, V comes out of text_emb which is conditioning variable
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            # device,
            1,
            guidance_scale > 1,
            None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        text_emb = prompt_embeds
        if guidance_scale > 1:
            text_emb = torch.cat([negative_prompt_embeds, prompt_embeds])
        

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, encoder_hidden_states=text_emb,
            ).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
