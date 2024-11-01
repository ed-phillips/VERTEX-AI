####### Example file for Generative AI analysis #########
import os
import pandas as pd
import numpy as np
import dspy
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# if we don't have a local model downloaded, get it from the hf hub

model_dir = "./local_model"
# model_name = "facebook/MobileLLM-350M"  # try new mobile llm
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name= "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"

# if not os.path.exists(model_dir):
#     try:
#         snapshot_download(repo_id=model_name, cache_dir=model_dir)
#         print(f"Model {model_name} downloaded successfully.")
#     except Exception as e:
#         print(f"Failed to download model {model_name}: {str(e)}")

# early release so still some teething issues - for now just access normally
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


class CustomHFModel:
    """Custom class as mobile models aren't compatible with the HFModel wrapper
    (which is deprecated anyway)"""
    def __init__(self, model, tokenizer, device=None, **kwargs):

        # Determine the device (default to GPU if available)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.kwargs = kwargs  # Store additional keyword arguments for generation settings

    def __call__(self, input_text, **generate_kwargs):
        # Combine class kwargs with any specific call kwargs for flexibility
        generation_params = self.kwargs | generate_kwargs

        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # Generate predictions
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_params, eos_token_id=None)

        # Decode output to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
# Instantiate the custom model wrapper with the loaded model and tokenizer
custom_model = CustomHFModel(model, tokenizer, temperature=0.7, max_new_tokens=2048)

# Define the signature for insight generation
class DataInsights(dspy.Signature):
    """Generate comprehensive analytical insights from dataset statistics"""
    
    descriptive_table = dspy.InputField(desc="Markdown table with basic statistics describing a tabular dataset")
    analysis_request = dspy.InputField(desc="Specific aspect of the data to analyze")
    insights = dspy.OutputField(desc="3-5 clear, specific insights about the data in bullet point form")


class AnalysisGenerator:
    def __init__(self, model_dir="./local_model"):
        """Initialize the DSPy model and configure it"""
        dspy.configure(lm=custom_model)
        self.generate_insights = dspy.Predict(DataInsights)

    def generate_analysis(self, df, analysis_type="general"):
        """Generate insights based on the data and analysis type"""
        
        # Analysis requests for different types
        analysis_requests = {
            "general": "Provide general insights about the population and outcomes",
            "demographics": "Focus on demographic patterns and their relationship with outcomes",
            "outcomes": "Analyze outcome patterns and potential contributing factors",
            "trends": "Identify notable trends and patterns in the data"
        }
        
        # Generate insights
        descriptive_table = df.to_markdown()
        response = self.generate_insights(
            descriptive_table=descriptive_table,
            analysis_request=analysis_requests.get(analysis_type, analysis_requests["general"])
        )
        
        return response
        

# try out
if __name__=="__main__":
    test_table = pd.read_csv("data/desc_table.csv")
    ag = AnalysisGenerator()
    insights = ag.generate_analysis(df=test_table)
    print(insights)

