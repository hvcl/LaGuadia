import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file.' )
    parser.add_argument( '--output_path', type=str, required=True, help='Path to save the output keyword bank (.pt).' )
    parser.add_argument( '--keyword_column', type=str, default='keywords', help='Name of the CSV column containing keywords to encode.' )
    parser.add_argument( '--batch_size', type=int, default=8, help='Batch size for keyword encoding.' )
    
    return parser.parse_args()

def load_keyword_bank(path: str):
    """
    Load keyword bank from the path.

    :param path: Directory path containing keyword bank files.
    :type path: str
    """
    keyword_bank = {}

    bank_paths = os.listdir(path)
    for bank_path in bank_paths:
        _bank = torch.load(os.path.join(path, bank_path))
        keyword_bank.update(_bank)

    return keyword_bank

def get_keyword_list(df: pd.DataFrame, target_column: str = 'keywords') -> list:
    _temp_keyword_list = sorted(list(set(','.join(df[target_column].to_list()).split(','))))
    _keyword_list = []
    for _key in _temp_keyword_list:
        _keyword_list.append(_key.lower().strip())
    return sorted(list(set(_keyword_list)))

def main():
    args = parse_args()
    
    df = pd.read_csv(args.csv_path)
    
    model = AutoModel.from_pretrained("google/medsiglip-448").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("google/medsiglip-448")

    texts = get_keyword_list(df)
    
    keyword_bank = {}

    bs = args.batch_size
    for text_chunk in tqdm([texts[i:i + bs] for i in range(0, len(texts), bs)]):
        inputs = tokenizer(text=text_chunk, padding="max_length", return_tensors="pt").to("cuda")

        with torch.no_grad():
            text_feat = model.get_text_features(**inputs)

            print(text_feat.shape)
            
            for idx, keyword in enumerate(text_chunk):
                keyword_bank[keyword] = text_feat[idx].cpu()
    
    _save_dir = os.path.join(args.output_path, 'keywords.pt')
    print(f'Saving Keyword bank -> ({_save_dir})')
    torch.save(keyword_bank, _save_dir)

if __name__ == "__main__":
    main()