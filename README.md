# research
# Unit-test 
## Installing Requirements and Running Tests
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the test suite:
    ```bash
    python tests/test_case.py --model_dir "jan-hq/Jan-Llama3-0708" \\
                              --num_rows 100 \\ 
    ```
## Test Configuration

- The test suite uses the following model and dataset:
- Model: "jan-hq/Jan-Llama3-0708"
- Tokenizer: "jan-hq/llama-3-sound-init"
- Dataset: "jan-hq/instruction-speech-conversation-test"

## What the Tests Cover

1. Output validation (non-empty, correct type)
2. Token ID validation
3. Output length checks
4. Input-output relevance using BLEU
5. Special token handling
6. Numerical stability (NaN checks)
7. Check if EOS token are unique and at the end of the generated ids

## Continuous Integration

- This test suite can be integrated into CI/CD pipelines.
- model download and inference can take significant time.