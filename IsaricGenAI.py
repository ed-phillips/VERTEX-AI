####### Example file for Generative AI analysis #########
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

class CustomHFModel:
    def __init__(self, model, tokenizer, device=None, **kwargs):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def generate_insights(self, data_description, analysis_type):
        # Construct a clear prompt
        prompt = f"""Given the following data statistics:

{data_description}

Please analyze this data with a focus on {analysis_type} and provide 3-5 specific insights.
Format your response as bullet points.

INSIGHTS:
"""
        
        # Set up generation parameters
        generation_params = {
            'max_new_tokens': 512,  # Reduced from 2048 to be more focused
            'temperature': 1,
            'do_sample': True,
            'num_return_sequences': 1,
            'pad_token_id': self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        generation_params.update(self.kwargs)

        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params
                )

            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract insights using split
            insights = generated_text.split("INSIGHTS:")[-1].strip()
            
            
            return insights
        
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return f"Error during generation: {str(e)}"



class AnalysisGenerator:
    def __init__(self, model_dir="./models"):
        """Initialize the model and tokenizer with local caching"""
        self.model_name = MODEL_NAME
        self.model_dir = os.path.join(model_dir, self.model_name.split('/')[-1])
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Check if model is already downloaded
        model_files = os.listdir(self.model_dir) if os.path.exists(self.model_dir) else []
        model_needs_download = len(model_files) == 0
        
        if model_needs_download:
            print(f"Downloading model {self.model_name} to {self.model_dir}...")
            try:
                snapshot_download(
                    repo_id=self.model_name,
                    local_dir=self.model_dir,
                    # ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"]  # Ignore unnecessary files
                )
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                raise
        else:
            print(f"Loading model from local directory: {self.model_dir}")
            
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                use_fast=False,
                local_files_only=True if not model_needs_download else False
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True if not model_needs_download else False
            )
            
            # Initialize custom model wrapper
            self.custom_model = CustomHFModel(self.model, self.tokenizer)
            
        except Exception as e:
            print(f"Error initializing model/tokenizer: {str(e)}")
            raise


    def generate_analysis(self, df, analysis_type="general"):
        """Generate insights based on the data and analysis type"""
        
        analysis_requests = {
            "general": "general patterns and insights",
            "demographics": "demographic patterns and their relationships with outcomes",
            "outcomes": "outcome patterns and potential contributing factors",
            "trends": "notable trends and patterns in the data"
        }
        
        try:
            # Convert DataFrame to a more readable format
            data_description = df.to_markdown(index=False)
            
            # Generate insights
            insights = self.custom_model.generate_insights(
                data_description,
                analysis_requests.get(analysis_type, analysis_requests["general"])
            )
            
            return insights
            
        except Exception as e:
            print(f"Error generating analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"

