import os
import hydra
import logging
import pandas as pd
import os.path as osp
from tqdm import tqdm

# troch
import torch
from torch.utils.data import DataLoader

# transformers
from transformers import AutoProcessor, AutoImageProcessor
from transformers import get_cosine_schedule_with_warmup

# CLAM (from trident)
from trident.segmentation_models import segmentation_model_factory

# Lab Model
from laguadia import LaGuadiaModel
from laguadia.datasets import LaGuadiaStage2Dataset
from preparing.keyword_utils import get_keyword_list, load_keyword_bank
from laguadia.utils import seed_everything
from preparing.keyword_utils import get_keyword_list

# CuPy
import cupy as cp
cp.get_default_memory_pool().set_limit(size=32 * 1024**3)

 
# Preventing error
if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False


seed_everything(1813)

@hydra.main(config_path = "configs", version_base = None)
def main(configs):
    print('Model Name :',configs.model_name)
    fold = int(input(f'Enter fold number (1~5): '))
    
    configs.model_name = configs.model_name + f'_fold{fold}'

    # Make save directory
    os.makedirs(os.path.join(configs.save_dir, configs.model_name) , exist_ok=True)
    
    # Logging
    log_path = os.path.join(configs.save_dir, configs.model_name, "train_log.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger(__name__)
    
    # ============= Loading Model =============
    # Using CLAM
    artifact_remover_model = segmentation_model_factory('grandqc_artifact', remove_penmarks_only=True)

    model = LaGuadiaModel(configs)
    model.cuda()
    
    # Trainable Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Meta-Teacher Model
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    dinov3_processor = AutoImageProcessor.from_pretrained(configs.pretrained_model_name)
    
    _load_pretrained_from = configs.proj_weights
    if _load_pretrained_from is None:
        _load_pretrained_from = configs.proj_weights_root + f'_fold{fold}/best_stage1_projectors.pth'
        
    logger.info(f'Loading teacher model projectors from : {_load_pretrained_from}')
    state_dict = torch.load(_load_pretrained_from, weights_only=False)
    model.load_stage1_state_dict(state_dict['state_dict'], strict=False)

    logger.info(f'-------------------- Model info ------------------')
    logger.info(f"Total parameters      : {total_params:,}")
    logger.info(f"Trainable parameters  : {trainable_params:,}")
    logger.info(f"Non-trainable params  : {total_params - trainable_params:,}")
    logger.info(f"Trainable ratio       : {100 * trainable_params / total_params:.2f}%")

    # LoRA Model Optimizer
    lora_params, head_params = [], []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            print(n)
            lora_params.append(p)
        elif "proj" in n:
            print(n)
            head_params.append(p)
        else:
            print(n)
            head_params.append(p)
    
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 2e-4, "weight_decay": 0.0},
            {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.999),
    )

    # ============= Loading Data =============
    
    if configs.get('csv', None) is not None:
        csv = pd.read_csv(configs.csv)
    else:
        csv = pd.read_csv('/workspace/home/Paris/kskim/encoder/csv/TCGA_THCA_keywords_filtered.csv')

    # Load Keyword Bank
    keyword_bank = load_keyword_bank(configs.keyword_bank_path)
    
    _train_val_keywords = set(get_keyword_list(csv[csv[f'fold_{fold}'] != 'test']))
    _test_keywords = set(get_keyword_list(csv[csv[f'fold_{fold}'] == 'test']))
    _test_only_keywords = _test_keywords - _train_val_keywords
    
    _bank_list = []
    for _key in keyword_bank.keys():
        if _key not in _test_only_keywords:
            _bank_list.append(keyword_bank[_key])
    
    _bank_items = torch.stack(_bank_list).cuda()
    
    
    # Loading Dataset
    train_dataset = LaGuadiaStage2Dataset(args = configs, dfs = csv, tokenizer = processor.tokenizer, fold = fold, split='train', keyword_bank = keyword_bank, artifact_remover_model = artifact_remover_model, dinov3_processor = dinov3_processor)
    train_loader = DataLoader(train_dataset, batch_size=int(configs.encoder_batch), num_workers=8, shuffle = True, pin_memory=True, persistent_workers=True)
    
    val_dataset = LaGuadiaStage2Dataset(args = configs, dfs = csv, tokenizer = processor.tokenizer, fold = fold, split='val', keyword_bank = keyword_bank, artifact_remover_model = artifact_remover_model, dinov3_processor = dinov3_processor)
    val_loader = DataLoader(val_dataset, batch_size=int(configs.encoder_batch), num_workers=8, shuffle = True, pin_memory=True, persistent_workers=True)
    
    logger.info(f'--------------------Data loaded------------------')
    logger.info(f'train length   : {len(train_loader)}')
    logger.info(F'val length     : {len(val_loader)}')
    
    # ============= Calculate Optimizer ============
    accumulation_steps = 4
    
    # Set Train Steps
    steps_per_epoch = len(train_loader) // accumulation_steps
    num_training_steps = configs.n_epochs * steps_per_epoch
    num_warmup_steps = int(0.05 * num_training_steps)
    
    logger.info(f'num_warmup_steps : {num_warmup_steps} \nnum_training_steps : {num_training_steps}')

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    best_loss = torch.inf
    
    # resume model
    strat_epoch = 0
    if configs.resume is not None:
        logger.info(f'Loading model from {configs.resume}')
        state_dict = torch.load(configs.resume, weights_only=False)
        
        strat_epoch = state_dict['epoch']
        model.load_state_dict(state_dict['state_dict'], strict=False)
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler = state_dict['scheduler']
        logger.info(f'Load complete')
        logger.info(f'Start Epoch : {strat_epoch}')
        
    if not os.path.isdir(osp.join(configs.save_dir, configs.model_name)):
        os.mkdir(osp.join(configs.save_dir, configs.model_name))
    
    for e in range(strat_epoch, configs.n_epochs):
        if e < strat_epoch:
            print(f'Skip epoch {e}')
            continue
        
        # Train
        model.train()
        loss_sum = 0
        cnt = 0
        optimizer.zero_grad()
        
        print(f'{e+1} / {configs.n_epochs} : {configs.model_name}')
        train_bar = tqdm(train_loader, desc=f'[{e+1} / {configs.n_epochs}]')
        for iter, inputs in enumerate(train_bar):               
            cnt += 1
            
            if configs.get('batch_mode', False):
                loss, _ = model.forward_stage2_features_batch(
                    keyword_bank=_bank_items,
                    **inputs,
                )
            else:
                loss, _ = model.forward_stage2_features_v2(
                    keyword_bank=_bank_items,
                    **inputs,
                )

            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            loss_sum += loss.item()

            if (iter + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            train_bar.set_postfix(LOSS=f'{loss_sum / cnt:.4f}')
            
            # Saving Every 2000 iter
            if (iter + 1) % 2000 == 0:
                state = {
                    'epoch' : e,
                    'iter' : iter,
                    'state_dict' : model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler
                }
                torch.save(state, os.path.join(configs.save_dir, configs.model_name + f'/current_iter_model.pth'))
            
        if cnt > 0:
            logger.info(f'[Epoch {e}] Train loss : {loss_sum / cnt:.4f}')
            
        # Validation
        loss_sum = 0
        with torch.inference_mode():
            model.eval()
            cnt = 0
            val_bar = tqdm(val_loader, desc=f'[{e+1} / {configs.n_epochs}]')
            for _, inputs in enumerate(val_bar):
                cnt += 1

                if configs.get('batch_mode', False):
                    loss, _ = model.forward_stage2_features_batch(
                        keyword_bank=_bank_items,
                        **inputs,
                    )
                else:
                    loss, _ = model.forward_stage2_features_v2(
                        keyword_bank=_bank_items,
                        **inputs,
                    )
                
                loss_sum += loss.item()
                val_bar.set_postfix(LOSS=f'{loss_sum / cnt:.4f}')
                
        logger.info(f'[Epoch {e}] Validation loss : {loss_sum / cnt:.4f}')
        
        # Save every Epoch
        state = {
            'epoch' : e,
            'state_dict' : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler
        }
        torch.save(state, os.path.join(configs.save_dir, configs.model_name + f'/current_model.pth'))
        
        if (loss_sum / cnt) < best_loss:
            logger.info(f'Best model saved {best_loss} -> {loss_sum / cnt:.4f}')
            best_loss = loss_sum / cnt

            torch.save(state, os.path.join(configs.save_dir, configs.model_name + f'/best_model.pth'))
    logger.info(f'Train {configs.model_name} model complete for {configs.n_epochs} epochs.')

if __name__ == "__main__":
    main()