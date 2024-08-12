import pytest

def pytest_addoption(parser):
    parser.addoption("--model_dir", type=str, default="jan-hq/Jan-Llama3-0708", help="Hugging Face model link or local_dir")
    parser.addoption("--max_length", type=int, default=1024, help="Maximum length of the output")
    parser.addoption("--data_dir", type=str, required=True, help="Hugging Face model repository link or Data path")
    parser.addoption("--cache_dir", type=str, default=".", help="Absolute path to save the model and dataset")
    parser.addoption("--mode", type=str, default="audio", help="Mode of the model (audio or text)")
    parser.addoption("--num_rows", type=int, default=5, help="Number of dataset rows to process")
    parser.addoption("--output_file", type=str, default="output/", help="Output file path")

@pytest.fixture(scope="session")
def custom_args(request):
    return {
        "model_dir": request.config.getoption("--model_dir"),
        "max_length": request.config.getoption("--max_length"),
        "data_dir": request.config.getoption("--data_dir"),
        "cache_dir": request.config.getoption("--cache_dir"),
        "mode": request.config.getoption("--mode"),
        "num_rows": request.config.getoption("--num_rows"),
        "output_file": request.config.getoption("--output_file"),
    }