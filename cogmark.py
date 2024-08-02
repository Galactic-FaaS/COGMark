import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
from safetensors.torch import load_file
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path, model_type='transformer', device='cpu'):
    """
    Load a model from either Transformers or a .safetensor file.
    
    Args:
    model_path (str): Path to the model or name of the Transformers model.
    model_type (str): Either 'transformer' or 'safetensor'.
    device (str): The device to load the model onto.

    Returns:
    model: The loaded model.
    tokenizer: The tokenizer (if applicable, else None).
    """
    if model_type == 'transformer':
        model = AutoModel.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    elif model_type == 'safetensor':
        # Assuming the .safetensor file contains the state dict of the model
        state_dict = load_file(model_path)
        # You'll need to define the model architecture here
        model = YourModelClass()  # Replace with your actual model class
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model, None
    else:
        raise ValueError("Invalid model_type. Choose 'transformer' or 'safetensor'.")

class COGMarkBenchmark:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def run_all_tests(self):
        results = {}
        results['perception'] = self.test_perception()
        results['learning'] = self.test_learning()
        results['reasoning'] = self.test_reasoning()
        results['planning'] = self.test_planning()
        results['language'] = self.test_language()
        results['emotion'] = self.test_emotion()
        results['social_cognition'] = self.test_social_cognition()
        results['meta_cognition'] = self.test_meta_cognition()
        results['creativity'] = self.test_creativity()
        results['memory'] = self.test_memory()
        results['transfer_learning'] = self.test_transfer_learning()
        results['multi_task'] = self.test_multi_task()
        return results

    def test_perception(self):
        visual_inputs = torch.rand(100, 384, dtype=torch.float64, device=self.device)
        auditory_inputs = torch.rand(100, 384, dtype=torch.float64, device=self.device)
        env_inputs = torch.rand(100, 10, dtype=torch.float64, device=self.device)

        outputs = []
        attentions = []
        for v, a, e in zip(visual_inputs, auditory_inputs, env_inputs):
            output = self.model(v.unsqueeze(0), a.unsqueeze(0), e.unsqueeze(0))
            outputs.append(output.last_hidden_state.mean(dim=1))
            attentions.append(output.attentions[-1].mean(dim=1) if output.attentions else torch.tensor([0.0]))

        outputs = torch.cat(outputs)
        attentions = torch.cat(attentions)

        output_validity = torch.all((outputs >= 0) & (outputs <= 1)).item()
        attention_validity = torch.allclose(attentions.sum(dim=1), torch.ones(100, device=self.device)).item()

        # Test for multi-modal integration
        visual_only = self.model(visual_inputs[0].unsqueeze(0), torch.zeros_like(auditory_inputs[0]).unsqueeze(0), env_inputs[0].unsqueeze(0)).last_hidden_state.mean(dim=1)
        auditory_only = self.model(torch.zeros_like(visual_inputs[0]).unsqueeze(0), auditory_inputs[0].unsqueeze(0), env_inputs[0].unsqueeze(0)).last_hidden_state.mean(dim=1)
        multi_modal = self.model(visual_inputs[0].unsqueeze(0), auditory_inputs[0].unsqueeze(0), env_inputs[0].unsqueeze(0)).last_hidden_state.mean(dim=1)

        integration_score = torch.norm(multi_modal - (visual_only + auditory_only) / 2).item()

        return {
            'output_validity': output_validity,
            'attention_validity': attention_validity,
            'multi_modal_integration': integration_score
        }

    def test_learning(self):
        env = gym.make('CartPole-v1')
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float64, device=self.device).unsqueeze(0)

        rewards = []
        for episode in range(100):
            total_reward = 0
            done = False
            while not done:
                action_probs = self.model(state, state, torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
                action = torch.argmax(action_probs).item()
                next_state, reward, done, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float64, device=self.device).unsqueeze(0)
                
                # Here you would typically update the model, but we'll skip that for this benchmark
                
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float64, device=self.device).unsqueeze(0)

        learning_curve = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        learning_speed = np.argmax(learning_curve > 195) if np.any(learning_curve > 195) else len(learning_curve)

        return {
            'final_performance': np.mean(rewards[-10:]),
            'learning_speed': learning_speed
        }

    def test_reasoning(self):
        # Syllogistic reasoning
        syllogisms = [
            ("All A are B. All B are C. Therefore, all A are C.", True),
            ("All A are B. Some B are C. Therefore, some A are C.", True),
            ("No A are B. Some C are B. Therefore, some C are not A.", True),
            ("All A are B. No B are C. Therefore, some A are C.", False),
        ]

        correct = 0
        for premise, true_conclusion in syllogisms:
            inputs = self.tokenizer(premise, return_tensors="pt").to(self.device)
            reasoning_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            model_conclusion = reasoning_output.mean().item() > 0.5
            if model_conclusion == true_conclusion:
                correct += 1

        # Analogical reasoning
        analogies = [
            ("hand is to glove as foot is to", "sock"),
            ("car is to road as train is to", "track"),
            ("fish is to water as bird is to", "air"),
        ]

        analogy_scores = []
        for premise, target in analogies:
            inputs = self.tokenizer(premise, return_tensors="pt").to(self.device)
            analogy_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            generated_answer = self.tokenizer.decode(torch.argmax(analogy_output, dim=-1))
            analogy_scores.append(sentence_bleu([target.split()], generated_answer.split()))

        return {
            'syllogistic_accuracy': correct / len(syllogisms),
            'analogical_reasoning': np.mean(analogy_scores)
        }

    def test_planning(self):
        env = gym.make('FrozenLake-v1', is_slippery=False, map_name="8x8")
        
        def evaluate_plan():
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 100

            while not done and steps < max_steps:
                state_tensor = torch.tensor([state], dtype=torch.float64, device=self.device)
                action_probs = self.model(state_tensor, state_tensor, torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
                action = torch.argmax(action_probs).item()

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                steps += 1

            return total_reward, steps

        num_episodes = 100
        rewards = []
        steps_list = []

        for _ in range(num_episodes):
            reward, steps = evaluate_plan()
            rewards.append(reward)
            steps_list.append(steps)

        return {
            'success_rate': np.mean(rewards),
            'average_steps': np.mean(steps_list),
            'efficiency': np.mean([r/s if s > 0 else 0 for r, s in zip(rewards, steps_list)])
        }

    def test_language(self):
        questions = [
            "What is the capital of France?",
            "Who wrote 'Romeo and Juliet'?",
            "What is the largest planet in our solar system?",
            "What year did World War II end?",
            "What is the chemical symbol for gold?"
        ]
        
        answers = [
            "Paris",
            "William Shakespeare",
            "Jupiter",
            "1945",
            "Au"
        ]

        bleu_scores = []
        meteor_scores = []
        perplexities = []

        for question, answer in zip(questions, answers):
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            language_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            generated_response = self.tokenizer.decode(torch.argmax(language_output, dim=-1))

            bleu_scores.append(sentence_bleu([answer.split()], generated_response.split()))
            meteor_scores.append(meteor_score([answer.split()], generated_response.split()))

            # Calculate perplexity using GPT-2
            gpt2_inputs = self.gpt2_tokenizer(generated_response, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gpt2_model(**gpt2_inputs, labels=gpt2_inputs["input_ids"])
            perplexities.append(torch.exp(outputs.loss).item())

        return {
            'average_bleu': np.mean(bleu_scores),
            'average_meteor': np.mean(meteor_scores),
            'average_perplexity': np.mean(perplexities)
        }

    def test_emotion(self):
        emotion_texts = [
            "I just won the lottery!",
            "My dog died yesterday.",
            "I'm so angry I could scream.",
            "I feel very calm and peaceful.",
            "I'm really anxious about my exam."
        ]
        
        emotion_labels = ['joy', 'sadness', 'anger', 'calm', 'anxiety']

        emotion_vectors = []
        for text in emotion_texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            emotion_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            emotion_vectors.append(emotion_output.cpu().numpy())

        emotion_vectors = np.array(emotion_vectors)

        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(emotion_vectors))

        # Calculate emotion differentiation score (higher is better)
        differentiation_score = np.mean(distances)

        # Calculate emotion granularity (lower is better)
        granularity_score = np.std(distances)

        return {
            'emotion_differentiation': differentiation_score,
            'emotion_granularity': granularity_score
        }

    def test_social_cognition(self):
        scenarios = [
            ("Alice thinks that Bob believes the cake is in the blue box, but it's actually in the red box.", 
             {"Alice_belief": "blue", "reality": "red"}),
            ("Charlie knows that Dana doesn't know that the surprise party is tonight.", 
             {"Charlie_knowledge": True, "Dana_knowledge": False}),
            ("Eve pretends to be happy to see Frank, even though she's actually upset with him.", 
             {"Eve_displayed_emotion": "happy", "Eve_true_emotion": "upset"})
        ]

        accuracy = 0
        for scenario, ground_truth in scenarios:
            inputs = self.tokenizer(scenario, return_tensors="pt").to(self.device)
            belief_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            
            # Assume the model outputs probabilities for different mental states
            # You'll need to adapt this part based on your model's actual output format
            predicted_mental_states = self.interpret_mental_states(belief_output)
            
            if self.compare_mental_states(predicted_mental_states, ground_truth):
                accuracy += 1

        return {
            'theory_of_mind_accuracy': accuracy / len(scenarios)
        }

    def interpret_mental_states(self, output):
        # This is a placeholder function. You'll need to implement this based on your model's output format.
        return {"state1": output[0].item(), "state2": output[1].item()}

    def compare_mental_states(self, predicted, ground_truth):
        # This is a placeholder function. You'll need to implement this based on your model's output format.
        return predicted == ground_truth

    def test_meta_cognition(self):
        easy_tasks = [torch.rand(384, dtype=torch.float64, device=self.device) for _ in range(10)]
        hard_tasks = [torch.rand(384, dtype=torch.float64, device=self.device) * 0.1 for _ in range(10)]  # More subtle patterns

        easy_confidences = []
        hard_confidences = []

        for easy, hard in zip(easy_tasks, hard_tasks):
            easy_confidence = self.model(easy.unsqueeze(0), easy.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
            hard_confidence = self.model(hard.unsqueeze(0), hard.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
            
            easy_confidences.append(easy_confidence.mean().item())
            hard_confidences.append(hard_confidence.mean().item())

        confidence_correlation = pearsonr(easy_confidences + hard_confidences, [1]*10 + [0]*10)[0]

        return {
            'meta_cognitive_accuracy': confidence_correlation
        }

    def test_creativity(self):
        prompts = [
            "Create a new animal by combining two existing animals.",
            "Invent a new sport that can be played in zero gravity.",
            "Design a new type of transportation for a city of the future."
        ]

        creativity_scores = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            creative_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            generated_idea = self.tokenizer.decode(torch.argmax(creative_output, dim=-1))

            # Use GPT-2 perplexity as a proxy for creativity (lower perplexity = more predictable = less creative)
            gpt2_inputs = self.gpt2_tokenizer(generated_idea, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gpt2_model(**gpt2_inputs, labels=gpt2_inputs["input_ids"])
            creativity_scores.append(torch.exp(outputs.loss).item())

        return {
            'average_creativity_score': np.mean(creativity_scores)
        }

    def test_memory(self):
        # Test working memory
        sequence_lengths = [5, 10, 15, 20]
        working_memory_scores = []

        for length in sequence_lengths:
            sequence = torch.randint(0, 10, (length,), device=self.device).float()
            sequence_input = self.tokenizer(' '.join(map(str, sequence.tolist())), return_tensors="pt").to(self.device)
            memory_output = self.model(**sequence_input).last_hidden_state
            recalled_sequence = torch.argmax(memory_output[:, :length], dim=-1)
            working_memory_scores.append(accuracy_score(sequence.cpu(), recalled_sequence.cpu()))

        # Test long-term memory
        facts = [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
            "The Earth orbits around the Sun.",
        ]

        for fact in facts:
            inputs = self.tokenizer(fact, return_tensors="pt").to(self.device)
            self.model(**inputs)  # Expose model to fact

        # Test recall after a "delay" (we'll just run some other computations)
        for _ in range(100):
            self.model(torch.rand(384, device=self.device).unsqueeze(0), 
                       torch.rand(384, device=self.device).unsqueeze(0), 
                       torch.zeros(10, device=self.device).unsqueeze(0))

        long_term_memory_scores = []
        for fact in facts:
            question = fact.split(' is ')[0] + " is?"
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            memory_output = self.model(**inputs).last_hidden_state.mean(dim=1)
            recalled_fact = self.tokenizer.decode(torch.argmax(memory_output, dim=-1))
            long_term_memory_scores.append(sentence_bleu([fact.split()], recalled_fact.split()))

        return {
            'working_memory_score': np.mean(working_memory_scores),
            'long_term_memory_score': np.mean(long_term_memory_scores)
        }

    def test_transfer_learning(self):
        # Train on task A
        task_a_data = [(torch.rand(384, device=self.device), torch.randint(0, 2, (1,), device=self.device)) for _ in range(1000)]
        for input_a, target_a in task_a_data[:800]:  # Use 800 for training
            output_a = self.model(input_a.unsqueeze(0), input_a.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
            loss = F.binary_cross_entropy(output_a.mean(), target_a.float())
            # Here you would typically update the model, but we'll skip that for this benchmark

        # Evaluate on task A
        task_a_accuracy = []
        for input_a, target_a in task_a_data[800:]:  # Use 200 for testing
            output_a = self.model(input_a.unsqueeze(0), input_a.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
            pred_a = (output_a.mean() > 0.5).long()
            task_a_accuracy.append((pred_a == target_a).float().item())

        # Train on task B (similar but slightly different)
        task_b_data = [(torch.rand(384, device=self.device), torch.randint(0, 2, (1,), device=self.device)) for _ in range(100)]
        for input_b, target_b in task_b_data[:80]:  # Use 80 for training
            output_b = self.model(input_b.unsqueeze(0), input_b.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
            loss = F.binary_cross_entropy(output_b.mean(), target_b.float())
            # Here you would typically update the model, but we'll skip that for this benchmark

        # Evaluate on task B
        task_b_accuracy = []
        for input_b, target_b in task_b_data[80:]:  # Use 20 for testing
            output_b = self.model(input_b.unsqueeze(0), input_b.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
            pred_b = (output_b.mean() > 0.5).long()
            task_b_accuracy.append((pred_b == target_b).float().item())

        return {
            'task_a_accuracy': np.mean(task_a_accuracy),
            'task_b_accuracy': np.mean(task_b_accuracy),
            'transfer_efficiency': np.mean(task_b_accuracy) / np.mean(task_a_accuracy)
        }

    def test_multi_task(self):
        tasks = [
            ('classification', [(torch.rand(384, device=self.device), torch.randint(0, 5, (1,), device=self.device)) for _ in range(100)]),
            ('regression', [(torch.rand(384, device=self.device), torch.rand(1, device=self.device)) for _ in range(100)]),
            ('generation', [(torch.rand(384, device=self.device), self.tokenizer("Generate a random sentence.", return_tensors="pt").input_ids.to(self.device)) for _ in range(100)])
        ]

        results = {}

        for task_name, task_data in tasks:
            task_performance = []
            for input_data, target in task_data[:80]:  # Use 80 for training
                output = self.model(input_data.unsqueeze(0), input_data.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
                if task_name == 'classification':
                    loss = F.cross_entropy(output, target.long())
                elif task_name == 'regression':
                    loss = F.mse_loss(output.mean(), target)
                else:  # generation
                    loss = F.cross_entropy(output, target.squeeze(0))
                # Here you would typically update the model, but we'll skip that for this benchmark

            for input_data, target in task_data[80:]:  # Use 20 for testing
                output = self.model(input_data.unsqueeze(0), input_data.unsqueeze(0), torch.zeros(10, device=self.device)).last_hidden_state.mean(dim=1)
                if task_name == 'classification':
                    accuracy = (output.argmax() == target).float().item()
                    task_performance.append(accuracy)
                elif task_name == 'regression':
                    mse = F.mse_loss(output.mean(), target).item()
                    task_performance.append(-mse)  # Negative MSE so higher is better
                else:  # generation
                    similarity = F.cosine_similarity(output, target.float(), dim=-1).item()
                    task_performance.append(similarity)

            results[f'{task_name}_performance'] = np.mean(task_performance)

        results['multi_task_score'] = np.mean(list(results.values()))
        return results

def plot_results(results):
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('COGMark: Comprehensive Cognitive Benchmark Results')
    
    for i, (test_name, test_results) in enumerate(results.items()):
        ax = axs[i // 3, i % 3]
        ax.bar(test_results.keys(), test_results.values())
        ax.set_title(test_name.capitalize())
        ax.set_xticklabels(test_results.keys(), rotation=45, ha='right')
        
    plt.tight_layout()
    plt.savefig('cogmark_results.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run COGMark: Comprehensive Cognitive Benchmark')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model or name of the Transformers model')
    parser.add_argument('--model_type', type=str, choices=['transformer', 'safetensor'], default='transformer', help='Type of the model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (e.g., "cpu", "cuda")')
    args = parser.parse_args()

    # Load the model
    model, tokenizer = load_model(args.model_path, args.model_type, args.device)

    # Run the benchmark
    benchmark = COGMarkBenchmark(model, tokenizer, args.device)
    results = benchmark.run_all_tests()

    # Print results
    print("COGMark Benchmark Results:")
    for test_name, test_results in results.items():
        print(f"\n{test_name.capitalize()} Test:")
        for metric, value in test_results.items():
            print(f"  {metric}: {value}")

    # Save results
    with open('cogmark_results.json', 'w') as f:
        json.dump(results, f)

    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()