import os
import glob

import sys

import numpy as np
from PIL import Image
from pathlib import Path

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmengine.structures import PixelData

from torchvision import transforms, tv_tensors
import torch.nn.functional as F
import torch
import torch.nn as nn

try:
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
except ImportError:
    AutoImageProcessor = None
    Mask2FormerForUniversalSegmentation = None

from open_clip import create_model, tokenizer
from huggingface_hub import hf_hub_download
from myutils import UnNormalize
from prompts.imagenet_template import openai_imagenet_template, get_prompt_templates, get_prompt_strategy


_SAM2_IMPORT_ERROR = None
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as exc:
    _SAM2_IMPORT_ERROR = exc

import logging


@MODELS.register_module(name='gumix_rs')
class GUMixRSSegmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, dino_type, name_path, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, slide_stride=112, slide_crop=336, instance_mask_path=None,
                 mask_generator=None, multi_scales=None, use_georsclip=True, bg_idx=0,
                 sam2_points_per_side=8, prompt_type='imagenet',
                 georsclip_checkpoint=None, georsclip_cache_dir='georsclip',
                 dino_checkpoint=None, dino_repo_dir=None,
                 sam2_checkpoint='sam2_hiera_large.pt'):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.multi_scales = multi_scales or [1.0]
        self.highres_prior_alpha = 0.1
        self.sam2_checkpoint = sam2_checkpoint
        get_prompt_templates(prompt_type)
        if mask_generator == 'sam2' and _SAM2_IMPORT_ERROR is not None:
            raise ImportError('SAM2 could not be imported. Install the dependencies '
                              'in requirements.txt, including hydra-core.') from _SAM2_IMPORT_ERROR

        # Load GeoRSCLIP (RS5M) weights from Hugging Face and inject them into an OpenCLIP backbone.
        if use_georsclip:
            if model_type in ["ViT-H/14", "ViT-H-14"]:
                geors_repo_id = "Zilun/GeoRSCLIP"
                geors_ckpt_filename = "ckpt/RS5M_ViT-H-14.pt"
            elif model_type in ["ViT-B/32", "ViT-B-32"]:
                geors_repo_id = "Zilun/GeoRSCLIP"
                geors_ckpt_filename = "ckpt/RS5M_ViT-B-32.pt"
            else:
                raise ValueError(
                    f"GeoRSCLIP currently provides weights only for ViT-B/32 and ViT-H/14; "
                    f"please set model_type to 'ViT-H/14' or 'ViT-B/32' (current: {model_type})."
                )

            local_matches = sorted(glob.glob(
                os.path.join(georsclip_cache_dir, "**", geors_ckpt_filename), recursive=True))
            if georsclip_checkpoint is not None:
                if not os.path.isfile(georsclip_checkpoint):
                    raise FileNotFoundError(f'GeoRSCLIP checkpoint not found: {georsclip_checkpoint}')
                geors_ckpt_path = georsclip_checkpoint
            elif local_matches:
                geors_ckpt_path = local_matches[0]
            else:
                geors_ckpt_path = hf_hub_download(
                    repo_id=geors_repo_id,
                    filename=geors_ckpt_filename,
                    cache_dir=georsclip_cache_dir,
                )
            print(f"[GeoRSCLIP] checkpoint: {geors_ckpt_path}")

            self.clip = create_model(model_type, pretrained=None, precision='fp16')

            geors_state = torch.load(geors_ckpt_path, map_location="cpu")
            if isinstance(geors_state, dict) and "state_dict" in geors_state:
                geors_state = geors_state["state_dict"]
            geors_state = {k.replace("module.", ""): v for k, v in geors_state.items()}
            load_msg = self.clip.load_state_dict(geors_state, strict=False)
            print(f"[GeoRSCLIP] Loaded {model_type} from {geors_ckpt_path}")
            print(load_msg)

        else:
            self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
            print(f"[open_clip] Loaded {model_type} with pretrained={clip_type} (no GeoRSCLIP)")

        self.clip.eval().to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer.tokenize



        # ---------------------- DINO backbone ----------------------
        self.dino_type = dino_type

        if dino_type in ["dinov3_sat", "dinov3_vitl16"]:
            repo_dir = dino_repo_dir or "dinov3"
            weights_path = dino_checkpoint or "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


            assert os.path.isdir(repo_dir), f"DINOv3 repo not found: {repo_dir}"
            assert os.path.isfile(weights_path), f"DINOv3 weights not found: {weights_path}"

            self.dino = torch.hub.load(
                repo_dir,
                'dinov3_vitl16',
                source='local',
                weights=weights_path,
            )

            self.dino.eval().to(device)
            for p in self.dino.parameters():
                p.requires_grad = False

            self.dino_feat_dim = self.dino.embed_dim      # ViT-L: 1024
            patch_size = self.dino.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            self.dino_patch_size = patch_size             # = 16

            self.dino_qkv_output = None
            self.dino.blocks[-1].attn.qkv.register_forward_hook(self._hook_fn_forward_qkv)

            self.dummy = nn.Linear(1, 1)

            self.unnorm = UnNormalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]
            )
            self.norm = transforms.Normalize(
                [0.430, 0.411, 0.296],
                [0.213, 0.156, 0.143]
            )

        elif dino_type in ["dino_vitb8", "dino1-b8", "dino1_b8"]:
            dino_repo_dir = dino_repo_dir or "dino_repo"
            ckpt_path = dino_checkpoint or os.path.join(dino_repo_dir, "dino_vitbase8_pretrain.pth")

            assert os.path.isdir(dino_repo_dir), f"DINO repo not found: {dino_repo_dir}"
            assert os.path.isfile(ckpt_path), f"DINO ViT-B/8 weights not found: {ckpt_path}"

            if dino_repo_dir not in sys.path:
                sys.path.append(dino_repo_dir)

            from vision_transformer import vit_base

            self.dino = vit_base(patch_size=8, num_classes=0)

            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "teacher" in state:
                state = state["teacher"]

            missing, unexpected = self.dino.load_state_dict(state, strict=False)
            print(f"[DINO v1] loaded from {ckpt_path}")
            if len(missing) > 0:
                print("[DINO v1] missing keys:", missing)
            if len(unexpected) > 0:
                print("[DINO v1] unexpected keys:", unexpected)

            self.dino.eval().to(device)
            for p in self.dino.parameters():
                p.requires_grad = False

            self.dino_feat_dim = self.dino.embed_dim      # ViT-B: 768
            patch_size = self.dino.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            self.dino_patch_size = patch_size             # = 8

            self.dino_qkv_output = None
            self.dino.blocks[-1].attn.qkv.register_forward_hook(self._hook_fn_forward_qkv)

            self.dummy = nn.Linear(1, 1)

            self.unnorm = UnNormalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]
            )
            self.norm = transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )

            self.dino_qkv_output = None
            self.dino.blocks[-1].attn.qkv.register_forward_hook(self._hook_fn_forward_qkv)

            self.dummy = nn.Linear(1, 1)

            self.unnorm = UnNormalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]
            )
            self.norm = transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )

        else:
            raise ValueError(f"Unknown dino_type: {dino_type}")

        self.prompt_type = prompt_type

        # Build CLIP text embeddings for the class vocabulary.
        self.generate_category_embeddings(name_path, device)

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.instance_mask_path = instance_mask_path
        self.device = device
        self.bg_idx = bg_idx
        self.sam2_points_per_side = sam2_points_per_side

        self.set_mask_generator(mask_generator)

    def _hook_fn_forward_qkv(self, module, input, output):
        self.dino_qkv_output = output

    def _compute_instance_complexity(self, masks, min_area: int = 20, eps: float = 1e-6):
        """
        Compute a per-pixel geometry-complexity map (perimeter / area) from instance masks.

        Args:
            masks: [B,1,H,W] or [B,H,W], integer tensor; 0 denotes background / no instance.
        Returns:
            complexity: [B,1,H,W] float32; each pixel stores the complexity of its instance.
                        Background / tiny instances are set to 0.
        """
        if masks is None:
            return None

        if masks.dim() == 4:
            masks_ = masks[:, 0]
        elif masks.dim() == 3:
            masks_ = masks
        else:
            raise ValueError(f"masks ndim must be 3 or 4, got {masks.dim()}")

        B, H, W = masks_.shape
        device = masks_.device
        complexity = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

        for b in range(B):
            inst_ids = torch.unique(masks_[b])
            inst_ids = inst_ids[(inst_ids > 0) & (inst_ids < 10000)]

            for inst_id in inst_ids:
                inst_mask = (masks_[b] == inst_id)  # [H,W] bool
                area = inst_mask.sum().float()
                if area < min_area:
                    continue

                ys, xs = inst_mask.nonzero(as_tuple=True)
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                region = inst_mask[y0:y1 + 1, x0:x1 + 1]  # [h,w] bool

                up = F.pad(region[:-1, :], (0, 0, 1, 0), value=False)
                down = F.pad(region[1:, :], (0, 0, 0, 1), value=False)
                left = F.pad(region[:, :-1], (1, 0, 0, 0), value=False)
                right = F.pad(region[:, 1:], (0, 1, 0, 0), value=False)

                interior = up & down & left & right
                boundary = region & (~interior)

                perimeter = boundary.sum().float()  # perimeter

                comp_val = perimeter / (area + eps)

                complexity[b, 0, y0:y1 + 1, x0:x1 + 1][region] = comp_val

        return complexity

    def _single_scale_test(self, img, instance_masks, batch_img_metas):
        seg_logits = self.forward_slide(img, instance_masks, batch_img_metas,
                                        self.slide_stride, self.slide_crop)
        return seg_logits

    @torch.inference_mode()
    def multi_scale_test(self, img, instance_masks, batch_img_metas):
        """
        Multi-scale routing driven by geometry complexity and predictive uncertainty:
        - For each pixel, compute a score per scale.
        - Select the best scale via argmax.
        - Use the logits from the selected scale for that pixel.
        """
        B, C, H, W = img.shape

        complexity_map = self._compute_instance_complexity(instance_masks)  # [B,1,H,W] or None
        if complexity_map is None:
            complexity_map = torch.zeros(
                (B, 1, H, W), device=img.device, dtype=img.dtype
            )
        else:
            complexity_map = complexity_map.to(dtype=img.dtype, device=img.device)

        logits_list = []
        uncert_list = []
        eps = 1e-6

        for s in self.multi_scales:
            if s == 1.0:
                img_s = img
                masks_s = instance_masks
                metas_s = batch_img_metas
            else:
                new_size = (int(H * s), int(W * s))

                img_s = F.interpolate(
                    img, size=new_size,
                    mode='bilinear', align_corners=False
                )

                masks_s = F.interpolate(
                    instance_masks.float(),
                    size=new_size,
                    mode='nearest'
                ).long()

                metas_s = []
                for m in batch_img_metas:
                    m2 = m.copy()
                    m2['img_shape'] = (new_size[0], new_size[1], 3)
                    m2['scale_factor'] = (s, s, s, s)
                    metas_s.append(m2)

            logits_s = self._single_scale_test(img_s, masks_s, metas_s)  # [B,C,h_s,w_s]
            logits_s = F.interpolate(
                logits_s, size=(H, W),
                mode='bilinear', align_corners=False
            )  # [B,C,H,W]

            logits_list.append(logits_s)

            prob_s = logits_s.softmax(dim=1)  # [B,C,H,W]
            entropy_s = -(prob_s * prob_s.clamp_min(eps).log()).sum(
                dim=1, keepdim=True
            )  # [B,1,H,W]
            uncert_list.append(entropy_s)

        if len(self.multi_scales) == 1:
            return logits_list[0]

        num_scales = len(self.multi_scales)

        def normalize_map(x):
            Bn, _, Hn, Wn = x.shape
            x_flat = x.view(Bn, -1)
            x_min = x_flat.min(dim=1, keepdim=True)[0].view(Bn, 1, 1, 1)
            x_max = x_flat.max(dim=1, keepdim=True)[0].view(Bn, 1, 1, 1)
            return (x - x_min) / (x_max - x_min + 1e-6)

        comp_norm = normalize_map(complexity_map)            # [B,1,H,W]
        ent_norm_list = [normalize_map(e) for e in uncert_list]  # list of [B,1,H,W]
        ent_norm_stack = torch.stack(ent_norm_list, dim=1)   # [B,S,1,H,W]

        scales_tensor = torch.tensor(
            self.multi_scales, dtype=img.dtype, device=img.device
        )  # [S]

        if num_scales > 1:
            s_min = scales_tensor.min()
            s_max = scales_tensor.max()
            if (s_max - s_min) < 1e-6:
                geo_pref = torch.zeros_like(scales_tensor)
            else:
                geo_pref = 2.0 * (scales_tensor - (s_min + s_max) / 2) / (s_max - s_min)
        else:
            geo_pref = torch.zeros_like(scales_tensor)

        geo_pref = geo_pref.view(1, num_scales, 1, 1, 1)

        kappa0 = comp_norm.view(B, -1).median(dim=1)[0].view(B, 1, 1, 1)
        comp_center = comp_norm - kappa0
        comp_expand = comp_center.unsqueeze(1).expand(B, num_scales, 1, H, W)

        geo_weight = 1.0  # complexity, geometry
        unc_weight = 1.0  # uncertainty

        score = geo_weight * geo_pref * comp_expand - unc_weight * ent_norm_stack

        logits_stack = torch.stack(logits_list, dim=1)      # [B,S,C,H,W]
        B_, S_, C_, H_, W_ = logits_stack.shape

        k = min(1, S_)
        tau = 1.0

        topk_score, topk_idx = torch.topk(score, k=k, dim=1)

        weights = F.softmax(topk_score / tau, dim=1)

        index_k = topk_idx.expand(-1, -1, C_, -1, -1)  # [B,k,C,H,W]
        cand_logits = torch.gather(logits_stack, dim=1, index=index_k)  # [B,k,C,H,W]

        weights_exp = weights.expand(-1, -1, C_, -1, -1)  # [B,k,C,H,W]
        fused_logits = (weights_exp * cand_logits).sum(dim=1)  # [B,C,H,W]



        return fused_logits


    def forward_feature(self, img, masks):
        if self.dino_type in ["dinov3_sat", "dinov3_vitl16"]:
            return self._forward_feature_dinov3(img, masks)
        elif self.dino_type in ["dino_vitb8", "dino1-b8", "dino1_b8"]:
            return self._forward_feature_dino1(img, masks)
        else:
            raise ValueError(f"Unknown dino_type: {self.dino_type}")

    @torch.inference_mode()
    def _forward_feature_dinov3(self, img, masks):
        if isinstance(img, list):
            img = img[0]

        img_clip = img  # upsample

        img_dino = F.interpolate(
            img, scale_factor=2.0, mode='bilinear', align_corners=False
        )

        imgs_norm = [self.norm(self.unnorm(img_dino[i])) for i in range(len(img_dino))]
        imgs_norm = torch.stack(imgs_norm, dim=0)  # [B, 3, H_dino, W_dino]

        dino_dtype = next(self.dino.parameters()).dtype
        imgs_norm = imgs_norm.to(
            dtype=dino_dtype,
            device=self.dino.patch_embed.proj.weight.device
        )

        feats = self.dino.get_intermediate_layers(imgs_norm, n=4)  # list of 4 tensors
        last_feat = feats[-1]
        B, T_all, C = last_feat.shape

        patch = self.dino_patch_size  # 16
        h_feat = img_dino.shape[-2] // patch
        w_feat = img_dino.shape[-1] // patch
        num_spatial = h_feat * w_feat

        complexity_img = self._compute_instance_complexity(masks)
        if complexity_img is None:
            complexity_img = torch.zeros(
                (B, 1, img_clip.shape[-2], img_clip.shape[-1]),
                device=img_clip.device, dtype=img_clip.dtype
            )
        else:
            complexity_img = complexity_img.to(
                dtype=img_clip.dtype, device=img_clip.device
            )

        complexity_dino = F.interpolate(
            complexity_img,
            size=(h_feat, w_feat),
            mode='bilinear',
            align_corners=False
        )
        complexity_flat = complexity_dino.reshape(B, 1, num_spatial)

        patch_list = []
        for f in feats:
            p = f[:, -num_spatial:, :]
            p = F.normalize(p, dim=-1)
            patch_list.append(p)

        patch_tokens = torch.stack(patch_list, dim=1)  # [B,4,HW,C]

        base_layer_weights = torch.tensor(
            [0.1, 0.2, 0.3, 0.4],
            device=patch_tokens.device,
            dtype=patch_tokens.dtype
        ).view(1, 4, 1)

        geo_sensitivity = torch.tensor(
            [0.5, 0.2, -0.2, -0.5],
            device=patch_tokens.device,
            dtype=patch_tokens.dtype
        ).view(1, 4, 1)

        comp = complexity_flat.clamp_min(0.0)
        layer_scores = base_layer_weights + geo_sensitivity * comp  # [B,4,HW]
        layer_weights = torch.softmax(layer_scores, dim=1).unsqueeze(-1)  # [B,4,HW,1]

        dino_feats = (patch_tokens * layer_weights).sum(dim=1)      # [B,HW,C]
        dino_feats = F.normalize(dino_feats, dim=-1)

        dino_feats = dino_feats.to(dtype=img_clip.half().dtype, device=img_clip.device)

        image_features = self.clip.encode_image(
            img_clip.half(),
            dino_feats=dino_feats,
            feat_shape=(h_feat, w_feat),
            instance_masks=masks,
        )

        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], h_feat, w_feat)

        logits = F.interpolate(logits, size=img_clip.shape[-2:], mode='bilinear')

        return logits

    @torch.inference_mode()
    def _forward_feature_dino1(self, img, masks):
        """
        DINO v1 ViT-B/8 variant:
        - No 2x upscaling; use the current crop resolution.
        - Use the last 4 blocks' patch tokens with geometry-aware layer weighting.
        - Keep the rest of the pipeline aligned with the DINOv3 branch.
        """
        if isinstance(img, list):
            img = img[0]

        img_clip = img

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)  # [B, 3, H, W]

        dino_param = next(self.dino.parameters())
        imgs_norm = imgs_norm.to(
            dtype=dino_param.dtype,
            device=dino_param.device
        )

        feats = self.dino.get_intermediate_layers(imgs_norm, n=4)  # list of 4 tensors
        last_feat = feats[-1]
        B, T_all, C = last_feat.shape  # [B, 1+HW, C]

        patch = self.dino_patch_size  # ViT-B/8: 8
        h_feat = img.shape[-2] // patch
        w_feat = img.shape[-1] // patch
        num_spatial = h_feat * w_feat  # HW

        complexity_img = self._compute_instance_complexity(masks)  # [B,1,H,W] or None
        if complexity_img is None:
            complexity_img = torch.zeros(
                (B, 1, img_clip.shape[-2], img_clip.shape[-1]),
                device=img_clip.device, dtype=img_clip.dtype
            )
        else:
            complexity_img = complexity_img.to(
                dtype=img_clip.dtype, device=img_clip.device
            )

        complexity_dino = F.interpolate(
            complexity_img,
            size=(h_feat, w_feat),
            mode='bilinear',
            align_corners=False
        )  # [B,1,h_feat,w_feat]
        complexity_flat = complexity_dino.reshape(B, 1, num_spatial)  # [B,1,HW]

        patch_list = []
        for f in feats:
            # DINO v1: token = [CLS] + HW patch
            p = f[:, -num_spatial:, :]      # [B, HW, C]
            p = F.normalize(p, dim=-1)
            patch_list.append(p)

        patch_tokens = torch.stack(patch_list, dim=1)

        base_layer_weights = torch.tensor(
            [0.1, 0.2, 0.3, 0.4],
            device=patch_tokens.device,
            dtype=patch_tokens.dtype
        ).view(1, 4, 1)  # [1,4,1]

        geo_sensitivity = torch.tensor(
            [0.5, 0.2, -0.2, -0.5],
            device=patch_tokens.device,
            dtype=patch_tokens.dtype
        ).view(1, 4, 1)  # [1,4,1]

        comp = complexity_flat.clamp_min(0.0)          # [B,1,HW]
        layer_scores = base_layer_weights + geo_sensitivity * comp  # [B,4,HW]
        layer_weights = torch.softmax(layer_scores, dim=1)          # [B,4,HW]
        layer_weights = layer_weights.unsqueeze(-1)                 # [B,4,HW,1]

        dino_feats = (patch_tokens * layer_weights).sum(dim=1)      # [B,HW,C]
        dino_feats = F.normalize(dino_feats, dim=-1)

        dino_feats = dino_feats.to(
            dtype=img_clip.half().dtype,
            device=img_clip.device
        )

        image_features = self.clip.encode_image(
            img_clip.half(),
            dino_feats=dino_feats,  # geometry
            feat_shape=(h_feat, w_feat),
            instance_masks=masks,
        )
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], h_feat, w_feat)

        logits = F.interpolate(logits, size=img_clip.shape[-2:], mode='bilinear')

        return logits
    def forward_slide(self, img, instance_masks, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_instance_masks = instance_masks[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                    crop_instance_masks = nn.functional.pad(crop_instance_masks, pad, value=10000)
                crop_seg_logit = self.forward_feature(crop_img, crop_instance_masks).detach()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        torch.cuda.empty_cache()

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is None:  # demo_gradio
            inputs, img_path = inputs
            batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]
            instance_masks = (self.generate_mask(img_path)).unsqueeze(0)
        else:  # evaluation
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
            instance_masks = [self.generate_mask(data_sample.img_path) for data_sample in data_samples]
            instance_masks = torch.stack(instance_masks, dim=0)

        self.instance_masks = instance_masks.int()
        instance_masks = F.interpolate(instance_masks.unsqueeze(1).float(), size=inputs.shape[2:], mode='nearest').int()

        if self.multi_scales is not None and not (len(self.multi_scales) == 1 and self.multi_scales[0] == 1.0):
            seg_logits = self.multi_scale_test(inputs, instance_masks, batch_img_metas)
        else:
            seg_logits = self._single_scale_test(inputs, instance_masks, batch_img_metas)

        H, W = seg_logits.shape[-2:]

        masks = self.instance_masks.float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # [B,1,H0,W0]
        elif masks.dim() == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)

        masks = F.interpolate(
            masks, size=(H, W),
            mode='nearest'
        )  # [B,1,H,W]

        self.instance_masks = masks.squeeze(1).int()

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logit = seg_logits[i] * self.logit_scale
            seg_logit = seg_logit.softmax(0)  # n_queries * h * w

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits_background = seg_logit[:num_queries - num_cls + 1]
                seg_logits_background = seg_logits_background.max(0, keepdim=True)[0]
                seg_logits_stuff = seg_logit[num_queries - num_cls + 1:]
                seg_logit = torch.cat([seg_logits_background, seg_logits_stuff])

            seg_pred = seg_logit.argmax(0, keepdim=True)
            seg_pred[seg_logit.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            # Map Correction
            mask_values = torch.unique(self.instance_masks[i])
            mask_values = mask_values[1:]
            masks = mask_values.unsqueeze(1).unsqueeze(1) == self.instance_masks[i].unsqueeze(0).expand(
                len(mask_values), -1, -1)
            masks = masks.unsqueeze(1)
            for mask in masks:
                seg_pred[mask] = torch.mode(seg_pred[mask])[0]
            # Optionally align predictions/logits to GT resolution (metric-safe).
            target_h, target_w = seg_pred.shape[-2:]
            if data_samples is not None and hasattr(data_samples[i], 'gt_sem_seg'):
                target_h, target_w = data_samples[i].gt_sem_seg.data.shape[-2:]
                pred_h, pred_w = seg_pred.shape[-2:]

                if (pred_h, pred_w) != (target_h, target_w):
                    seg_logit = F.interpolate(
                        seg_logit.unsqueeze(0),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

            pred = seg_pred.float()
            if pred.dim() == 3:  # [1, H, W] -> [1,1,H,W]
                pred = pred.unsqueeze(0)
            elif pred.dim() == 2:  # [H,W] -> [1,1,H,W]
                pred = pred.unsqueeze(0).unsqueeze(0)

            pred = F.interpolate(
                pred, size=(target_h, target_w),
                mode='nearest'
            )  # [1,1,target_h,target_w]

            seg_pred = pred.squeeze(0).long()

            if data_samples is None:  # demo_gradio
                return seg_pred
            else:  # evaluation
                data_samples[i].set_data({
                    'seg_logit':
                        PixelData(**{'data': seg_logit}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def generate_category_embeddings(self, name_path, device=torch.device('cuda')):
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        thin_keywords = [
            'road', 'street', 'highway', 'lane', 'path', 'trail',
            'rail', 'railway', 'track', 'runway', 'bridge',

            'river', 'canal', 'stream', 'waterway',
            'coast', 'shoreline', 'shore', 'bank', 'harbor',

            'boundary', 'edge', 'border',
        ]

        thin_query_ids = []
        for qid, name in enumerate(query_words):
            lower = name.lower()
            if any(kw in lower for kw in thin_keywords):
                thin_query_ids.append(qid)

        if len(thin_query_ids) > 0:
            self.thin_query_ids = torch.tensor(
                thin_query_ids, dtype=torch.long, device=device
            )
        else:
            self.thin_query_ids = None

        def encode_template_group(qw, templates):
            query = self.tokenizer([temp(qw) for temp in templates]).to(device)
            feature = self.clip.encode_text(query)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature /= feature.norm()
            return feature

        prompt_type_lower = (self.prompt_type or '').lower()
        prompt_alias_enabled = 'alias' in prompt_type_lower
        prompt_alias_blend_weight = 0.0
        if prompt_alias_enabled and 'blendalias' in prompt_type_lower:
            prompt_alias_blend_weight = 0.10
            for key, weight in (
                ('blendalias02', 0.02),
                ('blendalias03', 0.03),
                ('blendalias04', 0.04),
                ('blendalias05', 0.05),
                ('blendalias06', 0.06),
                ('blendalias07', 0.07),
                ('blendalias08', 0.08),
                ('blendalias10', 0.10),
                ('blendalias15', 0.15),
            ):
                if key in prompt_type_lower:
                    prompt_alias_blend_weight = weight
                    break

        def normalize_prompt_alias_name(qw):
            return ' '.join(qw.replace('_', ' ').strip().lower().split())

        prompt_alias_class_blend_weights = {}
        if prompt_alias_blend_weight > 0:
            if 'compound05' in prompt_type_lower:
                prompt_alias_class_blend_weights.update({
                    'road flooded': 0.05,
                    'building non-flooded': 0.05,
                })
            if 'compound04' in prompt_type_lower:
                prompt_alias_class_blend_weights.update({
                    'road flooded': 0.04,
                    'building non-flooded': 0.04,
                })
            if 'road05' in prompt_type_lower:
                prompt_alias_class_blend_weights['road flooded'] = 0.05
            if 'road04' in prompt_type_lower:
                prompt_alias_class_blend_weights['road flooded'] = 0.04
            if 'pool07' in prompt_type_lower:
                prompt_alias_class_blend_weights['pool'] = 0.07
            if 'pool10' in prompt_type_lower:
                prompt_alias_class_blend_weights['pool'] = 0.10

        def get_prompt_aliases(qw):
            if not prompt_alias_enabled:
                return [qw]

            normalized = normalize_prompt_alias_name(qw)
            alias_map = {
                'bareland,barren': ['bare land', 'barren land', 'exposed soil'],
                'bareland, barren': ['bare land', 'barren land', 'exposed soil'],
                'impervious surface': ['paved surface', 'asphalt or concrete surface', 'sealed ground surface'],
                'low vegetation': ['grass and low shrubs', 'low plants', 'short vegetation'],
                'clutter': ['miscellaneous clutter', 'unclassified clutter', 'small miscellaneous objects'],
                'building flooded': ['flooded building', 'inundated building', 'building surrounded by floodwater'],
                'building non-flooded': ['non-flooded building', 'dry building', 'building not affected by floodwater'],
                'road flooded': ['flooded road', 'inundated road', 'road covered by floodwater'],
                'road non-flooded': ['non-flooded road', 'dry road', 'road not covered by floodwater'],
                'pool': ['swimming pool', 'blue pool water'],
                'facade': ['building facade', 'vertical building wall', 'wall facade'],
                'other': ['other land-cover area', 'miscellaneous background area'],
            }

            aliases = [qw]
            aliases.extend(alias_map.get(normalized, []))
            if ',' in qw:
                aliases.extend(part.strip() for part in qw.split(',') if part.strip())

            unique_aliases = []
            seen = set()
            for alias in aliases:
                alias = ' '.join(alias.replace('_', ' ').strip().split())
                key = alias.lower()
                if alias and key not in seen:
                    seen.add(key)
                    unique_aliases.append(alias)
            return unique_aliases[:4]

        def encode_prompt_group(qw, templates):
            aliases = get_prompt_aliases(qw)
            base_feature = encode_template_group(aliases[0], templates)
            if len(aliases) == 1:
                return base_feature

            alias_features = [encode_template_group(alias, templates) for alias in aliases[1:]]
            if prompt_alias_blend_weight > 0:
                alias_feature = torch.stack(alias_features, dim=0).mean(dim=0)
                blend_weight = prompt_alias_class_blend_weights.get(
                    normalize_prompt_alias_name(aliases[0]), prompt_alias_blend_weight
                )
                feature = base_feature * (1.0 - blend_weight) + alias_feature * blend_weight
            else:
                feature = torch.stack([base_feature] + alias_features, dim=0).mean(dim=0)
            feature /= feature.norm()
            return feature

        prompt_strategy = get_prompt_strategy(self.prompt_type)
        if prompt_alias_enabled:
            if prompt_alias_blend_weight > 0:
                print(f'[Prompt] alias=blend, max_aliases=4, alias_weight={prompt_alias_blend_weight}')
                if prompt_alias_class_blend_weights:
                    class_weight_summary = ', '.join(
                        f'{name}:{weight}'
                        for name, weight in sorted(prompt_alias_class_blend_weights.items())
                    )
                    print(f'[Prompt] alias_class_weights={class_weight_summary}')
            else:
                print('[Prompt] alias=mean, max_aliases=4')
        if prompt_strategy is None:
            print(f"[Prompt] type={self.prompt_type}, mode=flat, templates={len(get_prompt_templates(self.prompt_type))}")
        else:
            summary = ', '.join(
                f"{name}:w={weight},n={len(templates)}"
                for name, weight, templates in prompt_strategy
            )
            print(f"[Prompt] type={self.prompt_type}, mode=weighted_groups, {summary}")

        query_features = []
        with torch.inference_mode():
            for qw in query_words:
                if prompt_strategy is None:
                    templates = get_prompt_templates(self.prompt_type)
                    feature = encode_prompt_group(qw, templates)
                else:
                    group_features = []
                    for _, weight, templates in prompt_strategy:
                        if weight <= 0 or len(templates) == 0:
                            continue
                        group_features.append(encode_prompt_group(qw, templates) * float(weight))
                    feature = torch.stack(group_features, dim=0).sum(dim=0)
                    feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

    def set_mask_generator(self, generator_type):
        self.mask_generator_type = generator_type
        if generator_type == 'eomt':
            from eomt.infer import get_eomt
            self.mask_generator = get_eomt(cfg_file="eomt_large_640.yaml", use_compile=True)  # use torch.compile
        elif generator_type == 'mask2former':
            if AutoImageProcessor is None or Mask2FormerForUniversalSegmentation is None:
                raise ImportError("transformers is required when mask_generator='mask2former'")
            self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
            self.mask_generator = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-coco-panoptic", device_map=self.device)
            logging.disable(logging.WARNING)
        elif generator_type == 'entityseg':
            from CropFormer.demo_mask2former.demo import get_entityseg
            self.confidence_threshold = 0.5
            self.mask_generator = get_entityseg(cfg_file="mask2former_hornet_3x.yaml",
                                                ckpt_path="Mask2Former_hornet_3x_576d0b.pth")
        elif generator_type == 'sam2':
            if _SAM2_IMPORT_ERROR is not None:
                raise ImportError('SAM2 dependencies are missing.') from _SAM2_IMPORT_ERROR
            if not os.path.isfile(self.sam2_checkpoint):
                raise FileNotFoundError(f'SAM2 checkpoint not found: {self.sam2_checkpoint}')
            sam2 = build_sam2("sam2_hiera_l.yaml", self.sam2_checkpoint, device=self.device,
                              apply_postprocessing=False)
            sam2 = sam2.half()
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=self.sam2_points_per_side,
                pred_iou_thresh=0.4,
                stability_score_thresh=0.4,
                multimask_output=False,
            )
            print(f"[SAM2] points_per_side={self.sam2_points_per_side}")

    def generate_mask(self, img_path):
        # output shape: [H, W];
        # type is int and the minimum denotes the union of unsegmented regions

        if self.mask_generator_type is None:
            # load pre-generated SAM2 masks
            instance_mask = np.load(os.path.join(self.instance_mask_path, Path(img_path).stem + '.npz'))[
                'instance_mask']
            instance_mask = torch.from_numpy(instance_mask).to(self.device)
        elif self.mask_generator_type == 'eomt':
            img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
            with torch.inference_mode(), torch.autocast(dtype=torch.float16, device_type="cuda"):
                imgs = [img.to(self.device)]
                img_sizes = [img.shape[-2:] for img in imgs]

                transformed_imgs = self.mask_generator.resize_and_pad_imgs_instance_panoptic(imgs)
                mask_logits_per_layer, class_logits_per_layer = self.mask_generator(transformed_imgs)
                mask_logits = F.interpolate(mask_logits_per_layer[-1], self.mask_generator.img_size, mode="bilinear")
                mask_logits = self.mask_generator.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)

                preds = self.mask_generator.to_per_pixel_preds_panoptic(
                    mask_logits,
                    class_logits_per_layer[-1],
                    self.mask_generator.stuff_classes,
                    self.mask_generator.mask_thresh,
                    self.mask_generator.overlap_thresh,
                )[0]

            instance_mask = preds[..., 1]
        elif self.mask_generator_type == 'mask2former':
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs = self.mask_generator(**inputs)

            result = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            instance_mask = result["segmentation"]
        elif self.mask_generator_type == 'entityseg':
            from detectron2.data.detection_utils import read_image
            img = read_image(img_path, format="BGR")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                predictions = self.mask_generator(img)

            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores

            selected_indexes = (pred_scores >= self.confidence_threshold)
            selected_scores = pred_scores[selected_indexes]
            selected_masks = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            instance_mask = torch.zeros((m_H, m_W), dtype=torch.int, device=self.device)

            selected_scores, ranks = torch.sort(selected_scores)
            ranks = ranks + 1
            for index in ranks:
                instance_mask[(selected_masks[index - 1] == 1)] = int(index)
        elif self.mask_generator_type == 'sam2':
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                masks = self.mask_generator.generate(image)
            instance_mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
            if len(masks) != 0:
                sorted_anns = sorted(masks, key=(lambda x: x['area']))  # predicted_iou
                instance_id = 1
                for ann in sorted_anns:
                    m = ann['segmentation']
                    instance_mask[m] = instance_id
                    instance_id += 1
            instance_mask = torch.from_numpy(instance_mask).to(self.device)
        else:
            instance_mask = None

        return instance_mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def _stack_batch_gt(self, batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
