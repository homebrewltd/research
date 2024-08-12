import pytest
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import time

@pytest.fixture(scope="module")
def model_setup(custom_args):
    args = custom_args
    model_name = args.model_dir.split("/")[-1]
    save_dir_output = f'{args.output_file}/{model_name}-{args.mode}-Result.csv'
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_length, skip_special_tokens=False)
    
    model_save_dir = os.path.join(args.cache_dir, args.model_dir)
    if not os.path.exists(model_save_dir):
        snapshot_download(args.model_dir, local_dir=model_save_dir, max_workers=64)
    else:
        print(f"Found {model_save_dir}. Skipping download.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_save_dir)
    llm = LLM(model_save_dir, tokenizer=model_save_dir, gpu_memory_utilization=-1)
    
    data_save_dir = os.path.join(args.cache_dir, args.data_dir)
    dataset = load_dataset(args.data_dir, split='train')
    num_rows = min(args.num_rows, len(dataset))
    
    return args, tokenizer, llm, dataset, num_rows, sampling_params, save_dir_output

@pytest.fixture(scope="module")
def inference_results(model_setup):
    args, tokenizer, llm, dataset, num_rows, sampling_params, _ = model_setup
    results = []
    
    def vllm_sound_inference(sample_id):
        sound_messages = dataset[sample_id]['sound_convo'][0]
        expected_output_messages = dataset[sample_id]['sound_convo'][1]
        sound_input_str = tokenizer.apply_chat_template([sound_messages], tokenize=False, add_generation_prompt=True)
        text_input_str = dataset[sample_id]['prompt']
        expected_output_str = tokenizer.apply_chat_template([expected_output_messages], tokenize=False)
        
        outputs = llm.generate(sound_input_str, sampling_params)
        output_based_on_sound = outputs[0].outputs[0].text
        output_token_ids = outputs[0].outputs[0].token_ids
        
        return text_input_str, output_based_on_sound, expected_output_str, output_token_ids
    
    def vllm_qna_inference(sample_id):
        text_input_str = dataset[sample_id]['prompt']
        expected_answer_str = dataset[sample_id]['answer']
        question_str = tokenizer.apply_chat_template([text_input_str], tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(question_str, sampling_params)
        output_based_on_question = outputs[0].outputs[0].text
        output_token_ids = outputs[0].outputs[0].token_ids
        
        return text_input_str, output_based_on_question, expected_answer_str, output_token_ids
    if args.mode == "audio":
        for i in range(num_rows):
            results.append(vllm_sound_inference(i))
    elif args.mode == "text":
        for i in range(num_rows):
            results.append(vllm_qna_inference(i))
    
    df_results = pd.DataFrame(results, columns=['input', 'output', 'expected_output', 'output_token_ids'])
    df_results.to_csv(save_dir_output, index=False, encoding='utf-8')
    print(f"Successfully saved in {save_dir_output}")
    
    return results

def test_model_output(inference_results):
    for text_input_str, output_based_on_sound, expected_output_str, output_token_ids in inference_results:
        assert len(output_based_on_sound) > 0, "Output should not be empty"
        assert isinstance(output_based_on_sound, str), "Output should be a string"
        assert all(token >= 0 for token in output_token_ids), "Output tokens should be valid"

def test_special_tokens(model_setup, inference_results):
    _, tokenizer, _, _, _, _, _ = model_setup
    special_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]
    for token in special_tokens:
        if token:
            encoded = tokenizer.encode(token)
            assert encoded[0] != -100, f"Special token {token} should not be ignored"

def test_no_nan_outputs(inference_results):
    for _, output, _, _ in inference_results:
        assert not any(np.isnan(float(word)) for word in output.split() if word.replace('.', '').isdigit()), "Output should not contain NaN values"

def test_eos_token_generation(model_setup, inference_results):
    _, tokenizer, _, _, _, _, _ = model_setup
    eos_token_id = tokenizer.eos_token_id
    for _, _, _, output_token_ids in inference_results:
        assert eos_token_id in output_token_ids, "EOS token not found in the generated output"
        assert output_token_ids[-1] == eos_token_id, "EOS token is not at the end of the sequence"
        assert output_token_ids.count(eos_token_id) == 1, f"Expected 1 EOS token, but found {output_token_ids.count(eos_token_id)}"

# Additional tests can be added here