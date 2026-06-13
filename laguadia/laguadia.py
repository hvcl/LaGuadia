import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from torchvision.transforms import v2

# PEFT
from peft import get_peft_model, LoraConfig

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

# Utils
def get_similarity_boundary(_anchor: torch.Tensor, _key_embeds: torch.Tensor) -> torch.Tensor:
    _sim = F.cosine_similarity(_anchor, _key_embeds, dim=1)
    return min(_sim)

def get_mean_similarity_boundary(_anchor: torch.Tensor, _key_embeds: torch.Tensor) -> torch.Tensor:
    _sim = F.cosine_similarity(_anchor, _key_embeds, dim=1)
    return torch.mean(_sim)

def get_negative_pairs(_anchor: torch.Tensor, _bank_items: torch.Tensor, _min_sim: torch.Tensor = torch.Tensor([0.5]), n: int = 5) -> torch.Tensor:
    _sim = F.cosine_similarity(_anchor, _bank_items, dim=1)

    _idx = torch.where(_sim < _min_sim)[0]
    _sampled_idx = _idx[torch.randperm(_idx.numel())[:n]]
    return _bank_items[_sampled_idx]

def cosine_distill_per_sample(pred, target, eps=1e-8):
    pred = F.normalize(pred, dim=-1, eps=eps)
    target = F.normalize(target.detach(), dim=-1, eps=eps)
    return 1.0 - (pred * target).sum(dim=-1)

class DistillProjector(nn.Module):
    def __init__(self, dim_s: int, dim_t: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim_s)
        self.proj = nn.Linear(dim_s, dim_t)
        
    def forward(self, cls):
        return self.proj(self.ln(cls))

class LaGuadiaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.verbose = args.verbose
        self.text_dim = args.embed_dim_contrast # Vision-Language Distance Dim
        self.intermediate_dim = args.intermediate_dim
        self.inference_mode = args.inference_mode
        self.stage = args.stage
        self.negative_n = args.negative_n
        self.projector_only = args.get('projector_only', False)
        self.extraction_mode = args.get('extraction', False)
        
        # Teacher encoders
        self.teachers = []
        if self.args.use_gigapath:
            # Projection layer
            self.gigapath_proj = nn.Linear(1536, self.intermediate_dim)
            self.gigapath_proj.apply(self._init_weights)
            self.teachers.append('GigaPath')
        
        if self.args.use_uni:
            # Projection layer
            self.uni_proj = nn.Linear(1024, self.intermediate_dim)
            self.uni_proj.apply(self._init_weights)
            self.teachers.append('UNI-h')
        
        if self.args.use_virchow2:
            # Projection layer
            self.virchow2_proj = nn.Linear(1280, self.intermediate_dim)
            self.virchow2_proj.apply(self._init_weights)
            self.teachers.append('Virchow2')
        
        # Guide Langauge Model (Meta-Teacher)
        if not self.extraction_mode:
            self.print(f'Load Language Model from google/medsiglip-448')
            self.meta_teacher = AutoModel.from_pretrained("google/medsiglip-448") # 1152
            freeze_model(self.meta_teacher)
    
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        
        self.tau = 0.05 # Tau for weight SoftMax
        
        self.print(f'Using {len(self.teachers)} teachers. ({", ".join(self.teachers)})')

        if self.stage > 2 or self.extraction_mode:
            # PEFT Config
            peft_config = LoraConfig(
                inference_mode=self.inference_mode,
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )

            self.print('Init student DINOv3 encoder with LoRA PEFT...')
            student = AutoModel.from_pretrained(
                args.pretrained_model_name,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )

            self.student_hidden_size = student.config.hidden_size

            freeze_model(student)
            self.student = get_peft_model(student, peft_config)
            
            if self.extraction_mode: 
                self.print('[LaGuadia] Model created for feature extraction mode')
            
            # Distillation Projector
            self.dist_giagpath_proj = DistillProjector(self.student_hidden_size, 1536) 
            self.dist_uni_proj = DistillProjector(self.student_hidden_size, 1024)
            self.dist_virchow2_proj = DistillProjector(self.student_hidden_size, 1280)
            
            self.dist_giagpath_proj.apply(self._init_weights)
            self.dist_uni_proj.apply(self._init_weights)
            self.dist_virchow2_proj.apply(self._init_weights)

        if self.stage > 2 and not self.extraction_mode:
            self.print('Freeze Teacher projector')
            freeze_model(self.gigapath_proj)
            freeze_model(self.uni_proj)
            freeze_model(self.virchow2_proj)
        
        self._init_models()

    def _init_models(self):
        self.print('Initialize logit scale and bias...')
        torch.nn.init.zeros_(self.logit_scale)
        torch.nn.init.zeros_(self.logit_bias)

    def forward(
        self, 
        dinov3_images: torch.Tensor, 
        **kwargs,
    ):
        """
        Feature extraction forward pass for LaGuardia.
        
        Args:
            dinov3_images: (B, 3, H, W) input image tensor.
            **kwargs: ignored; kept for interface compatibility.
        
        Returns:
            (B, D) CLS token embeddings from the student encoder.
        """
        
        dinov3_images = dinov3_images.cuda()
        feats = self.student(dinov3_images, **kwargs).last_hidden_state
        return feats[:,0]

    def calculate_logits_per_image(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor, scale: bool = True):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
        
        if scale:
            logit_scale, logit_bias = self.meta_teacher.logit_scale.to(text_embeds.device), self.meta_teacher.logit_bias.to(text_embeds.device)
            logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        
        logits_per_image = logits_per_text.t()
        return logits_per_image

    # Training model with pre-extracted feature
    def forward_stage1(
        self,
        texts,
        medgemma_feats: torch.Tensor,
        keyword_bank: torch.Tensor,
        uni_feats: torch.Tensor = None,
        gigapath_feats: torch.Tensor = None,
        virchow2_feats: torch.Tensor = None,
        nt_idxs: torch.Tensor = None,
        **kwargs,
    ):
        B, K, L = texts.shape
        medgemma_feats = medgemma_feats.cuda() # MedSigLIP Feature
        
        uni_feats = uni_feats.cuda()
        gigapath_feats = gigapath_feats.cuda()
        virchow2_feats = virchow2_feats.cuda()
        
        texts = texts.cuda() # Keywords
        
        # Pahse 1: Get Logits per images
        pseudo_target = []
        _text_features = []

        non_tunmor_index = nt_idxs
        
        with torch.inference_mode():           
            # Passing Keyword from bank
            text_embeds = texts.cuda()
            
            for i in range(B):
                image_embeds = medgemma_feats[i]
                
                logits_per_image = self.calculate_logits_per_image(image_embeds=image_embeds, text_embeds=text_embeds[i, :non_tunmor_index[i]])
                probs = torch.softmax(logits_per_image, dim=0)
                _pseudo_target = torch.argmax(probs, dim=0) # [1] : tensor([2], device='cuda:0')
                
                _text_features.append(text_embeds[i])
                pseudo_target.append(_pseudo_target)
        
        # Phase 2: Select closest sample
        negative_samples = []
        positive_samples = []
        
        _bank_items = keyword_bank

        # Original
        for i in range(B):
            _selected_idx = pseudo_target[i]
            
            # Check if non-tumor is selected
            if _selected_idx == non_tunmor_index[i]:
                _negative_sample = get_negative_pairs(_text_features[i][_selected_idx], _bank_items, n=self.negative_n)
            else:
                _min_sim = get_mean_similarity_boundary(_text_features[i][_selected_idx], _text_features[i][:-1])
                _negative_sample = get_negative_pairs(_text_features[i][_selected_idx], _bank_items, _min_sim, n=self.negative_n)

            negative_samples.append(_negative_sample)
            positive_samples.append(_text_features[i][_selected_idx])
        
        negative_samples = torch.stack(negative_samples, dim=0)
        positive_samples = torch.stack(positive_samples, dim=0)
        target_samples = torch.cat([positive_samples.unsqueeze(1), negative_samples], dim=1) # .to(gigapath_feats.device) # [B, 1 + n, text_dim]
        
        # Pahse 3 : Select Tumor vs Non-Tumor
        # Project vision features to vision-language common space
        if gigapath_feats is not None: gigapath_inter_feats = self.gigapath_proj(gigapath_feats) # [B, inter_dim]
        if uni_feats is not None: uni_inter_feats = self.uni_proj(uni_feats)
        if virchow2_feats is not None: virchow2_inter_feats = self.virchow2_proj(virchow2_feats)
        
        # Contrastive Learning (SigLIP)
        gigapath_loss, uni_loss, virchow2_loss = torch.Tensor(0), torch.Tensor(0), torch.Tensor(0)
        if gigapath_feats is not None:
            gigapath_logits = self.get_batch_logits(target_samples, gigapath_inter_feats.unsqueeze(1))
            gigapath_loss = self.siglip_loss(gigapath_logits)

        if uni_feats is not None:
            uni_logits = self.get_batch_logits(target_samples, uni_inter_feats.unsqueeze(1))
            uni_loss = self.siglip_loss(uni_logits)

        if virchow2_feats is not None:
            virchow2_loss = self.get_batch_logits(target_samples, virchow2_inter_feats.unsqueeze(1))
            virchow2_loss = self.siglip_loss(virchow2_loss)

        return gigapath_loss + uni_loss + virchow2_loss

    def forward_stage2(
        self,
        texts: torch.Tensor,
        dinov3_images: torch.Tensor,
        uni_feats: torch.Tensor = None,
        gigapath_feats: torch.Tensor = None,
        virchow2_feats: torch.Tensor = None,
        nt_idxs: torch.Tensor = None,
        **kwargs,
    ):
        B = texts.shape[0]
        
        texts = texts.cuda() # Keywords
        dinov3_images = dinov3_images.cuda() # Students Image

        uni_feats = uni_feats.cuda()
        gigapath_feats = gigapath_feats.cuda()
        virchow2_feats = virchow2_feats.cuda()
        
        # a : Calculate teacher weight
        with torch.inference_mode():
            gigapath_embeds = self.gigapath_proj(gigapath_feats)
            uni_embeds = self.uni_proj(uni_feats)
            virchow2_embeds = self.virchow2_proj(virchow2_feats)

            weights = []
            
            for i in range(B):
                # Teacher A. (GigaPath)
                gigapath_lpi = self.calculate_logits_per_image(image_embeds=gigapath_embeds[i], text_embeds=texts[i, :nt_idxs[i]], scale=False)
                
                # Teacher B. (UNI)
                uni_lpi = self.calculate_logits_per_image(image_embeds=uni_embeds[i], text_embeds=texts[i, :nt_idxs[i]], scale=False)
                
                # Teacher C. (Virchow 2)
                virchow2_lpi = self.calculate_logits_per_image(image_embeds=virchow2_embeds[i], text_embeds=texts[i, :nt_idxs[i]], scale=False)
                
                # Soft Voting
                overall_lpi = gigapath_lpi + uni_lpi + virchow2_lpi
                probs = torch.softmax(overall_lpi, dim=0)
                _voted_target = torch.argmax(probs, dim=0)
                
                _teacher_similarlitery = torch.Tensor([gigapath_lpi[_voted_target], uni_lpi[_voted_target], virchow2_lpi[_voted_target]])
                _weight = F.softmax(_teacher_similarlitery / self.tau, dim=0)
                
                weights.append(_weight)
        
        weights = torch.vstack(weights).cuda() # [B, n_teacher]
        
        # b : CLS token Distillation
        feats = self.student(dinov3_images).last_hidden_state
        
        dist_cls_1 = self.dist_giagpath_proj(feats[:, 0])
        dist_cls_2 = self.dist_uni_proj(feats[:, 0])
        dist_cls_3 = self.dist_virchow2_proj(feats[:, 0])
        
        l1 = cosine_distill_per_sample(dist_cls_1, gigapath_feats)
        l2 = cosine_distill_per_sample(dist_cls_2, uni_feats)
        l3 = cosine_distill_per_sample(dist_cls_3, virchow2_feats)
        
        w1, w2, w3 = weights[:,0], weights[:,1], weights[:,2] # Split Weights
        
        # Calculate Loss
        loss = (w1 * l1 + w2 * l2 + w3 * l3).mean()
        
        return loss, (l1.mean().item(), l2.mean().item(), l3.mean().item())

    def forward_ablation(
        self,
        texts: torch.Tensor,
        dinov3_images: torch.Tensor,
        uni_feats: torch.Tensor = None,
        gigapath_feats: torch.Tensor = None,
        virchow2_feats: torch.Tensor = None,
        **kwargs,
    ):
        B = texts.shape[0]

        uni_feats = uni_feats.cuda()
        gigapath_feats = gigapath_feats.cuda()
        virchow2_feats = virchow2_feats.cuda()
        dinov3_images = dinov3_images.cuda()
        texts = texts.cuda()

        # a: CLS token based Distillation
        feats = self.student(dinov3_images).last_hidden_state
        
        dist_cls_1 = self.dist_giagpath_proj(feats[:, 0])
        dist_cls_2 = self.dist_uni_proj(feats[:, 0])
        dist_cls_3 = self.dist_virchow2_proj(feats[:, 0])
        
        l1 = cosine_distill_per_sample(dist_cls_1, gigapath_feats)
        l2 = cosine_distill_per_sample(dist_cls_2, uni_feats)
        l3 = cosine_distill_per_sample(dist_cls_3, virchow2_feats)
        
        loss = ((l1 + l2 + l3) / 3).mean()
        return loss, (l1.mean().item(), l2.mean().item(), l3.mean().item())

    def siglip_loss(self, batch_logits: torch.Tensor):
        targets = torch.zeros_like(batch_logits).to(batch_logits.device)
        targets[:, 0] = 1
        
        m1_diag1 = -torch.ones_like(batch_logits) + 2 * targets
        loglik = torch.nn.functional.logsigmoid(m1_diag1 * batch_logits)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        return loss
    
    def get_batch_logits(self, meta_image_feats: torch.Tensor, teacher_feats: torch.Tensor):
        # L2 normalize
        image_embeds = meta_image_feats / meta_image_feats.norm(p=2, dim=-1, keepdim=True)
        text_embeds  = teacher_feats  / teacher_feats.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.bmm(text_embeds, image_embeds.transpose(1, 2))

        # scale & bias
        logits_per_text = logits_per_text * self.logit_scale.exp() + self.logit_bias

        # (B, N, T)
        logits_per_image = logits_per_text.transpose(1, 2).squeeze(1)  # (B, T, N) -> (B, N)
        return logits_per_image

    def stage1_state_dict(self):
        _state_keys = ["gigapath_proj", "uni_proj","virchow2_proj", "logit_scale", "logit_bias"]
        
        _filtered_state = {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(name) for name in _state_keys)
        }
        
        return _filtered_state
    
    def load_stage1_state_dict(self, state_dict, strict=False):
        return self.load_state_dict(state_dict, strict=strict)
    
    def print(self, text: int, level: int = 1):
        if self.verbose >= level:
            print(text)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)