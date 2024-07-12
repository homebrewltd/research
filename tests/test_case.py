import unittest
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on a Sound-To-Text Model.")
    parser.add_argument("--model_dir", type=str, default="jan-hq/Jan-Llama3-0708", help="Hugging Face model link or local_dir")
    parser.add_argument("--model_save_dir", type=str, required=False, help="Local directory that model is saved")
    parser.add_argument("--data_dir", type=str, required=True, help="Hugging Face model repository link or Data path")
    parser.add_argument("--mode", type=str, default="audio", help="Mode of the model (audio or text)")
    parser.add_argument("--num_rows", type=int, default=5, help="Number of dataset rows to process")
    parser.add_argument("--output_file", type=str, default="output/", help="Output file path")
    return parser.parse_args()

class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = parse_arguments()
        # Global variables
        model_name = args.model_dir.split("/")[-1]
        cls.save_dir_output = f'{args.output_file}/{model_name}-{args.mode}-Result.csv'
        if not os.path.exists(args.output_file):
            os.makedirs(args.output_file)
        cls.sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, skip_special_tokens=False)
        model_dir = ""
        if args.model_save_dir:
            model_dir = args.model_save_dir
        else:
            # Download model
            if not os.path.exists(args.model_dir):
                snapshot_download(args.model_dir, local_dir=args.model_dir, max_workers=64)
            else:
                print(f"Found {args.model_dir}. Skipping download.")
            model_dir = args.model_dir
        # Model loading using vllm
        cls.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        cls.llm = LLM(model_dir, tokenizer=model_dir)
        
        # Load dataset
        cls.dataset = load_dataset(args.data_dir, cache_dir=".cache/")['train']
        cls.num_rows = min(args.num_rows, len(cls.dataset))
        cls.inference_results = []
        if args.mode == "audio":
            for i in range(cls.num_rows):
                cls.inference_results.append(cls.vllm_sound_inference(i))
        elif args.mode == "text":
            for i in range(cls.num_rows):
                cls.inference_results.append(cls.vllm_qna_inference(i))
        # print(cls.inference_results[0])
        df_results = pd.DataFrame(cls.inference_results, columns=['input', 'output', 'expected_output', 'output_token_ids'])
        df_results.to_csv(cls.save_dir_output, index=False, encoding='utf-8')
        print(f"Successfully saved in {cls.save_dir_output}")
    @classmethod
    def vllm_sound_inference(self, sample_id):
        sound_messages = self.dataset[sample_id]['sound_convo'][0]
        expected_output_messages = self.dataset[sample_id]['sound_convo'][1]
        sound_input_str = self.tokenizer.apply_chat_template([sound_messages], tokenize=False, add_generation_prompt=True)
        text_input_str = self.dataset[sample_id]['prompt']
        expected_output_str = self.tokenizer.apply_chat_template([expected_output_messages], tokenize=False)
        
        outputs = self.llm.generate(sound_input_str, self.sampling_params)
        output_based_on_sound = outputs[0].outputs[0].text
        output_token_ids = outputs[0].outputs[0].token_ids
        
        return text_input_str, output_based_on_sound, expected_output_str, output_token_ids
    @classmethod
    def vllm_qna_inference(self, sample_id):
        text_input_str = self.dataset[sample_id]['prompt']
        expected_answer_str = self.dataset[sample_id]['answer']
        question_str = self.tokenizer.apply_chat_template([text_input_str], tokenize=False, add_generation_prompt=True)
        
        outputs = self.llm.generate(question_str, self.sampling_params)
        output_based_on_question = outputs[0].outputs[0].text
        output_token_ids = outputs[0].outputs[0].token_ids
        
        return text_input_str, output_based_on_question, expected_answer_str, output_token_ids
    # @classmethod
    # def hf_inference(self, sample_id):
    #     question = self.dataset[sample_id]['question']
        
        
    #     return input_str, output_based_on_input, expected_output_str, output_token_ids
    def test_model_output(self):
        for text_input_str, output_based_on_sound, expected_output_str, output_token_ids in self.inference_results:
            # Test 1: Check if output is not empty
            self.assertGreater(len(output_based_on_sound), 0)
            
            # Test 2: Check if output is a string
            self.assertIsInstance(output_based_on_sound, str)
            
            # Test 3: Check if output tokens are valid (not containing -1 or other invalid tokens)
            self.assertTrue(all(token >= 0 for token in output_token_ids))
            
            # Test 4: Check if output is somewhat related to input (using simple word overlap)
            # reference_words = set(expected_output_str.lower().split())
            # output_words = set(output_based_on_sound.lower().split())
            # relevance_score = corpus_bleu(output_words, reference_words)
            # self.assertGreater(relevance_score, 0.3)

    def test_special_tokens(self):
        # Test 5: Check if special tokens are handled correctly
        special_tokens = [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]
        for token in special_tokens:
            if token:
                encoded = self.tokenizer.encode(token)
                self.assertNotEqual(encoded[0], -100)  # Special token should not be ignored
                # self.assertNotEqual(encoded[0], encoded[2])

    # def test_model_consistency(self):
    #     # Test 6: Check if model gives consistent outputs for the same input
    #     results = [self.inference_results[0][1] for _ in range(3)]  
    #     self.assertEqual(results[0], results[1])
    #     self.assertEqual(results[1], results[2])

    def test_no_nan_outputs(self):
        # Test 7: Check for NaN outputs
        for _, output, _, _ in self.inference_results:
            self.assertFalse(any(np.isnan(float(word)) for word in output.split() if word.replace('.', '').isdigit()))

    def test_eos_token_generation(self):
        # Test 8: Check if EOS token is generated
        for _, output_based_on_sound, _, output_token_ids in self.inference_results:
            eos_token_id = self.tokenizer.eos_token_id
            
            # Check if EOS token is in the output
            self.assertIn(eos_token_id, output_token_ids, "EOS token not found in the generated output")
            
            # Check if EOS token is at the end of the sequence
            self.assertEqual(output_token_ids[-1], eos_token_id, "EOS token is not at the end of the sequence")
            
            # Check if there's only one EOS token in the output
            eos_count = output_token_ids.count(eos_token_id)
            self.assertEqual(eos_count, 1, f"Expected 1 EOS token, but found {eos_count}")
        

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)