# COGMark

## Overview

COGMark is a comprehensive tool designed to evaluate the cognitive capabilities of large language models or cognitive architectures. It assesses various aspects of cognition, including perception, learning, reasoning, planning, language processing, emotion understanding, social cognition, meta-cognition, creativity, memory, transfer learning, and multi-task performance.

## Requirements

- Python 3.7+\
- PyTorch\
- Transformers\
- SafeTensors\
- Gym\
- NLTK\
- Matplotlib\
- SciPy\
- NumPy\
- Scikit-learn

You can install the required packages using:

```\
pip install torch transformers safetensors gym nltk matplotlib scipy numpy scikit-learn\
```

## Usage

To run the benchmark:

```\
python cogmark.py --model_path <model_path> --model_type <model_type> --device <device>\
```

Arguments:\
- `--model_path`: Path to the model or name of the Transformers model (required)\
- `--model_type`: Type of the model, either 'transformer' or 'safetensor' (default: 'transformer')\
- `--device`: Device to run the model on, e.g., "cpu" or "cuda" (default: "cuda" if available, else "cpu")

Example:\
```\
python cogmark.py --model_path "gpt2" --model_type transformer --device cuda\
```

## Benchmark Tests

The benchmark includes the following tests:

1\. **Perception Test**: Evaluates the model's ability to process multi-modal inputs and produce valid outputs and attention distributions.

2\. **Learning Test**: Assesses the model's capacity to learn and improve performance on a reinforcement learning task (CartPole environment).

3\. **Reasoning Test**: Tests both syllogistic and analogical reasoning capabilities.

4\. **Planning Test**: Evaluates the model's ability to navigate a complex environment (8x8 FrozenLake).

5\. **Language Test**: Assesses language understanding and generation using BLEU, METEOR, and perplexity metrics.

6\. **Emotion Test**: Measures the model's ability to differentiate between emotions and its emotional granularity.

7\. **Social Cognition Test**: Evaluates theory of mind capabilities using complex scenarios.

8\. **Meta-Cognition Test**: Assesses the model's ability to accurately gauge its own confidence.

9\. **Creativity Test**: Uses GPT-2 perplexity as a proxy for measuring the creativity of generated ideas.

10\. **Memory Test**: Evaluates both working memory and long-term memory capabilities.

11\. **Transfer Learning Test**: Measures the model's ability to transfer knowledge from one task to a related task.

12\. **Multi-task Test**: Assesses the model's performance on multiple types of tasks simultaneously.

## Output

The benchmark produces three types of output:

1\. **Console Output**: Prints the results of each test to the console.

2\. **JSON File**: Saves detailed results in a JSON file named 'enhanced_benchmark_results.json'.

3\. **Plot**: Generates a bar plot visualization of the results, saved as 'enhanced_benchmark_results.png'.

## Code Structure

- `load_model()`: Loads a model from either Transformers or a SafeTensor file.\
- `cogmark`: Main class containing all benchmark tests.\
- `plot_results()`: Generates a visualization of the benchmark results.\
- `main()`: Parses arguments, loads the model, runs the benchmark, and saves/plots results.

## Customization

To use this benchmark with custom models:

1\. Ensure your model has a similar interface to Hugging Face Transformers models.\
2\. If using a SafeTensor model, modify the `load_model()` function to correctly instantiate your model architecture.\
3\. Adjust the input/output processing in each test method to match your model's requirements.

## Limitations

- The benchmark assumes a certain model structure and output format. Modifications may be necessary for significantly different architectures.\
- Some tests (e.g., reinforcement learning) do not include actual model updates to keep the benchmark non-destructive.\
- The creativity and language generation evaluations use GPT-2 as a baseline, which may not be suitable for all use cases.

## Contributing

Contributions to improve or extend the benchmark are welcome. Please submit a pull request or open an issue for discussion.

## License

Apache 2.0
