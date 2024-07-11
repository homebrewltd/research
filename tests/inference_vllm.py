from accelerate.utils import merge_fsdp_weights
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
# Global variable
save_dir_output = 'output/Jan-Llama3-0708-Result.csv'
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
local_dir = "jan-hq/Jan-Llama3-0708"
snapshot_download("jan-hq/Jan-Llama3-0708", local_dir=local_dir, max_workers=64)
# Model loading using vllm
tokenizer = AutoTokenizer.from_pretrained("jan-hq/llama-3-sound-init")
llm = LLM(local_dir, tokenizer="jan-hq/llama-3-sound-init")
dataset = load_dataset("jan-hq/instruction-speech-conversation-test", cache_dir="/.cache/")['train']
def vllm_inference(sample_id):
    sound_messages = dataset[sample_id]['sound_convo'][0]
    expected_output_messages = dataset[sample_id]['sound_convo'][1]

    sound_input_str = tokenizer.apply_chat_template([sound_messages], tokenize=False, add_generation_prompt=True)
    text_input_str = dataset[sample_id]['prompt']
    output_based_on_text = tokenizer.apply_chat_template([expected_output_messages], tokenize=False)

    outputs = llm.generate(sound_input_str, sampling_params)
    output_based_on_text = outputs[0].outputs[0].text
    output_token_ids = outputs[0].outputs[0].token_ids

    print("-"*50)
    print("Text input: ", text_input_str)
    print("-"*50)
    print("Text output: ", output_str)
    print("-"*50)
    print("Expected output: ", expected_output_str)
    print("-"*50)
    print("Output token ids: ", output_token_ids)
    print("-"*50)

    return text_input_str, output_based_on_sound, output_based_on_text, output_token_ids
ouput_df = pd.DataFrame()
for i in range(len(dataset)):
    text_input_str, output_based_on_sound, output_based_on_text, output_token_ids = vllm_inference(i)
    # add to dictionary
    output_df['text_input'].append(text_input_str)
    output_df['output_based_on_sound'].append(output_based_on_sound)
    output_df['output_based_on_text'].append(output_based_on_text)
    output_df['output_token_ids'].append(output_token_ids)
output_df.to_csv(save_dir_output, index=False, encoding='utf-8')