import os
import json
import re
import torch
import random
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    T5ForConditionalGeneration, 
    T5Tokenizer
)
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import BLEU
import warnings

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

class CCQAPipeline:
    """
    CCQA Pipeline
    """
    def __init__(
        self, 
        slm_model_name: str = "model_name_or_path",  # Path to SLM model
        t5_model_path: str = "./finetuned_t5_model",  # Path to fine-tuned T5 model
        device: str = None
    ):
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Initializing CCQA Pipeline on device: {self.device}")

        print(f"Loading SLM model: {slm_model_name}")
        self.slm_tokenizer = AutoTokenizer.from_pretrained(slm_model_name)
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            slm_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        print(f"Loading T5 model: {t5_model_path}")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(self.device)
        
        print("Loading sentence transformer for similarity calculation")
        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        
        self.bleu_scorer = BLEU(effective_order=True)
        
        # CCQA parameters
        self.alpha = 0.4  # Weight for BLEU score  
        self.beta = 0.6   # Weight for cosine similarity
        
    def extract_numerical_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from model response"""
        if not response:
            return None
            
        patterns = [
            r'the answer is (?:[$€£¥₩+\-±×÷=≈])?\s*([0-9][0-9,]*(?:\.\d+)?)'
        ]
        
        text = response.lower()
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                answer = match.group(1).strip()
                if ',' in answer and answer.replace(',', '').isdigit():
                    answer = answer.replace(',', '')
                return answer
        return None
    
    def generate_solutions(self, question: str, n_solutions: int = 5) -> List[str]:
        """Generate multiple solutions using SLM with CoT prompting"""
        
        # GSM8K CoT prompt with few-shot examples
        cot_prompt = f"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
Q: {question} A:"""
        
        solutions = []
        for i in range(n_solutions):
            inputs = self.slm_tokenizer(cot_prompt, return_tensors="pt").to(self.device)
            
            # Set seed for reproducibility
            torch.manual_seed(42 + i)
            
            with torch.no_grad():
                outputs = self.slm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.slm_tokenizer.eos_token_id,
                )
            
            response = self.slm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            solutions.append(response.strip())
            
        return solutions
    
    def check_lcv_condition(self, answers: List[str], n_solutions: int) -> bool:
        """
        Check if Low Confidence Voting condition is met
        LCV condition from paper: max_j freq(A_j) < ⌈N/2⌉
        """
        if not answers:
            return True
            
        # Count frequency of each answer
        answer_counts = Counter(ans for ans in answers if ans is not None)
        if not answer_counts:
            return True
            
        # Get the most frequent answer
        max_count = answer_counts.most_common(1)[0][1]
        
        # LCV condition: max frequency < ceil(N/2)
        return max_count < (n_solutions + 1) // 2
    
    def generate_question_from_solution(self, solution: str) -> str:
        """Generate question from solution using fine-tuned T5"""
        
        # Prompt for T5 question generation
        input_text = f"""CRITICAL: Do not change ANY numeric values in the answer. 
Every number must be preserved EXACTLY in your question. 
Generate a question that would have this as its answer: {solution}"""
        
        input_ids = self.t5_tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.t5_model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=0.0
            )
        
        generated_question = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_question
    
    def calculate_similarity(self, original_question: str, generated_question: str) -> float:  
        # BLEU score (lexical similarity)
        bleu_score = self.bleu_scorer.sentence_score(
            generated_question, [original_question]
        ).score / 100.0  # Normalize to 0-1 range
        
        # Cosine similarity (semantic similarity) using Sentence-BERT
        embeddings = self.sentence_model.encode([original_question, generated_question])
        cosine_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # Combined similarity score
        combined_score = self.alpha * bleu_score + self.beta * cosine_sim
        return combined_score
    
    def ccqa_inference(self, question: str, n_solutions: int = 5) -> Dict[str, Any]:
        """
        Main CCQA inference pipeline
        """
        print(f"Processing question: {question[:50]}...")
        
        # Step 1: Generate multiple solutions
        print("Step 1: Generating multiple solutions...")
        solutions = self.generate_solutions(question, n_solutions)
        
        # Step 2: Extract answers from solutions
        print("Step 2: Extracting answers...")
        extracted_answers = []
        for i, solution in enumerate(solutions):
            answer = self.extract_numerical_answer(solution)
            extracted_answers.append(answer)
            print(f"  Solution {i+1}: {answer}")
        
        # Step 3: Check for majority voting
        print("Step 3: Checking LCV condition...")
        lcv_condition = self.check_lcv_condition(extracted_answers, n_solutions)
        
        if not lcv_condition:
            # Majority voting - select most frequent answer
            answer_counts = Counter(ans for ans in extracted_answers if ans is not None)
            if answer_counts:
                final_answer = answer_counts.most_common(1)[0][0]
                selected_solution_idx = extracted_answers.index(final_answer)
                method_used = "majority_voting"
                print(f"Majority voting selected: {final_answer}")
            else:
                final_answer = None
                selected_solution_idx = 0
                method_used = "no_valid_answer"
        else:
            # LCV condition met - use CCQA method
            print("Step 4: LCV condition met, applying CCQA...")
            
            # Generate questions for each solution
            generated_questions = []
            similarity_scores = []
            
            for i, solution in enumerate(solutions):
                if extracted_answers[i] is not None:
                    print(f"  Generating question for solution {i+1}...")
                    gen_question = self.generate_question_from_solution(solution)
                    generated_questions.append(gen_question)
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(question, gen_question)
                    similarity_scores.append(similarity)
                    print(f"    Similarity: {similarity:.4f}")
                else:
                    generated_questions.append("")
                    similarity_scores.append(0.0)
            
            # Select solution with highest similarity
            if similarity_scores and max(similarity_scores) > 0:
                selected_solution_idx = similarity_scores.index(max(similarity_scores))
                final_answer = extracted_answers[selected_solution_idx]
                method_used = "ccqa_similarity"
                print(f"CCQA selected solution {selected_solution_idx+1} with similarity {max(similarity_scores):.4f}")
            else:
                selected_solution_idx = 0
                final_answer = extracted_answers[0] if extracted_answers else None
                method_used = "fallback_first"
        
        return {
            "question": question,
            "solutions": solutions,
            "extracted_answers": extracted_answers,
            "lcv_condition": lcv_condition,
            "method_used": method_used,
            "selected_solution_idx": selected_solution_idx,
            "final_answer": final_answer,
            "generated_questions": generated_questions if lcv_condition else [],
            "similarity_scores": similarity_scores if lcv_condition else []
        }

def main():
    # Initialize CCQA pipeline
    ccqa = CCQAPipeline(
        slm_model_name="model_name_or_path",  # Update with actual SLM model path
        t5_model_path="./finetuned_t5_model"  # Update with actual path
    )
    
    # Example GSM8K question
    test_question = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    # Run CCQA inference
    result = ccqa.ccqa_inference(test_question, n_solutions=5)
    
    # Print results
    print("\n" + "="*80)
    print("CCQA INFERENCE RESULTS")
    print("="*80)
    print(f"Question: {result['question']}")
    print(f"LCV Condition: {result['lcv_condition']}")
    print(f"Method Used: {result['method_used']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Selected Solution Index: {result['selected_solution_idx']}")
    
    if result['lcv_condition']:
        print("\nSimilarity Scores:")
        for i, score in enumerate(result['similarity_scores']):
            if score > 0:
                print(f"  Solution {i+1}: {score:.4f}")
    
    print("\nSelected Solution:")
    print(result['solutions'][result['selected_solution_idx']])

if __name__ == "__main__":
    main()