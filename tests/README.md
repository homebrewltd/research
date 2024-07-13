---
datasets:
- jan-hq/instruction-speech-v1
language:
- en
license: apache-2.0
tags:
- sound language model
---

## Model Details

We have developed and released the family [Jan-Llama3](https://huggingface.co/collections/jan-hq/jan-llama3-668e4dad446c8736208dca4f). This family is natively understanding audio and text input.

We continue to expand [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) with sound understanding capabilities by leveraging 700M tokens [Instruction Speech v1](https://huggingface.co/datasets/jan-hq/instruction-speech-v1) dataset.

**Model developers** Homebrew Research.

**Input** Text and sound.

**Output** Text.

**Model Architecture** Llama-3.

**Language(s):** English.

## Intended Use

**Intended Use Cases** This family is primarily intended for research applications. This version aims to further improve the LLM on sound understanding capabilities.

**Out-of-scope** The use of Llama-3-Sound in any manner that violates applicable laws or regulations is strictly prohibited.

## How to Get Started with the Model

First, we need to convert the audio file to sound tokens

```python
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

def audio_to_sound_tokens(audio_path, target_bandwidth=1.5, device="cuda"):
    # Initialize Encodec
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(target_bandwidth)
    model.to(device)

    # Load and preprocess audio
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)

    # Encode audio
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

    # Flatten codes
    audio_code1, audio_code2 = codes[0][0], codes[0][1]
    flatten_tokens = torch.stack((audio_code1, audio_code2), dim=1).flatten().tolist()

    # Convert to sound tokens
    result = ''.join(f'<|sound_{num}|>' for num in flatten_tokens)
    return f'<|sound_start|>{result}<|sound_end|>'

# Usage
sound_tokens = audio_to_sound_tokens("/path/to/your/audio/file")
```

Then, we can inference the model the same as any other LLM.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def setup_pipeline(model_path, use_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = {"device_map": "auto"}
    
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_text(pipe, messages, max_new_tokens=64, temperature=0.0, do_sample=False):
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "temperature": temperature,
        "do_sample": do_sample,
    }

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Usage
llm_path = "jan-hq/Jan-Llama3-0708"
pipe = setup_pipeline(llm_path, use_4bit=True)
messages = [
    {"role": "user", "content": sound_tokens},
]
generated_text = generate_text(pipe, messages)
print(generated_text)
```

## Training process
**Training Metrics Image**: Below is a snapshot of the training loss curve visualized.

![train_loss_curve/png](https://cdn-uploads.huggingface.co/production/uploads/65713d70f56f9538679e5a56/9bv-kpnqrTxaBhiYrVHN7.png)

### Hardware

**GPU Configuration**: Cluster of 8x NVIDIA H100-SXM-80GB.
**GPU Usage**:
  - **Continual Training**: 8 hours.

### Training Arguments

| Parameter                  | Continual Training      | 
|----------------------------|-------------------------|
| **Epoch**                  | 1                       | 
| **Global batch size**      | 128                     | 
| **Learning Rate**          | 5e-5                    | 
| **Learning Scheduler**     | Cosine with warmup      | 
| **Optimizer**              | [Adam-mini](https://arxiv.org/abs/2406.16793)               | 
| **Warmup Ratio**           | 0.1                     | 
| **Weight Decay**           | 0.01                    |
| **beta1**                  | 0.9                     | 
| **beta2**                  | 0.98                    | 
| **epsilon**                | 1e-6                    | 
| **Gradient Cliping**       | 1.0                     | 

###
 Accelerate FSDP Config

```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: true
fsdp_config:
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Examples

1. Good example:
```
# Question:
Does the following review have a positive or negative opinion of the movie?

i thought the relationships were wonderful , 
Options: A). negative; B). positive;

# Expected Output:
The review has a B). positive opinion about the movie because they think the relationships were wonderful.

# Model's Output:
The review is positive.
```

2. Misunderstanding example:
```
# Question:
Translate the following sentence to Russian:
6 Work in JOBS and careers after 40

Russian:

# Expected Output:
6 Работать в РАБОТАХ и карьере после 40

Russian (Cyrillic): 6 Работать в РАБОТАХ и карьере после 40

Let me explain this to you like you're five years old: This sentence means that people can still work and have careers, even if they're older than 40 years old. The words are in a different language, called Russian.
# Model's Output:
В случае, когда вы хотите перевести фразу "The first step is to identify the language of the text." на русский язык, вы можете использовать следующую фразу: "Первый шаг - определить язык текста."
```

3. Off-tracked example:
```
# Quesion:
Answer the following question: -  The roots of the growing plants start to break up the rock - The plant acids dissolve the rock - The rock is broken down into smaller pieces - Erosion begins.  What might be the first step of the process?
Answer:

# Expected Output:
The first step of the process might be: The roots of the growing plants start to break up the rock.

# Model's Output:
Step 1: Identify the key elements in the premise and hypothesis.

Premise: "A man in a blue shirt and a red hat is holding a sign."
Key elements: man, blue shirt, red hat, holding, sign

Hypothesis: "A man is holding a sign."
Key elements: man, holding, sign

Step 2: Compare the key elements in the premise and hypothesis.

In the premise, we have a man wearing a blue shirt and a red hat, while in the hypothesis, we only have a man. The premise also mentions that the man is holding a sign, which is also mentioned in the hypothesis.

Step 3: Determine if the hypothesis is entailed by the premise.

Since the hypothesis states that a man is holding a sign, and the premise confirms that a man is holding a sign, we can conclude that the hypothesis is entailed by the premise. The additional information about the man's clothing in the premise does not contradict or negate the
```

Despite being undertrained, the model demonstrates an emerging grasp of sound-text semantics.


## Citation Information

**BibTeX:**

```
@article{Llama-3-Sound: Sound Instruction LLM 2024,
  title={Llama-3-Sound},
  author={Homebrew Research},
  year=2024,
  month=July},
  url={https://huggingface.co/jan-hq/Jan-Llama3-0708}
```

## Acknowledgement

- **[WhisperSpeech](https://github.com/collabora/WhisperSpeech)**

- **[Encodec](https://github.com/facebookresearch/encodec)**

- **[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)**