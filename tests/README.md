# research
# Unit-test 
## Installing Requirements and Running Tests
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the test suite:
    ```bash
    python test_case.py --model_dir "jan-hq/Jan-Llama3-0708" \\
                        --max_length 1024 \\
                        --data_dir "jan-hq/instruction-speech-conversation-test" \\
                        --mode "audio" \\
                        --num_rows 5 \\ 
    ```
## Test Configuration

- The test suite uses the following model and dataset:
- Model: "jan-hq/Jan-Llama3-0708"
- Tokenizer: "jan-hq/llama-3-sound-init"
- Dataset: "jan-hq/instruction-speech-conversation-test"

## What the Tests Cover

1. Output validation (non-empty, correct type)
2. Token ID validation
3. Input-output relevance using BLEU
4. Special token handling
5. Numerical stability (NaN checks)
6. Check if EOS token are unique and at the end of the generated ids

## Continuous Integration

- This test suite can be integrated into CI/CD pipelines.
- model download and inference can take significant time.