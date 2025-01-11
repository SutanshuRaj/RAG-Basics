from dotenv import load_dotenv
import os

class Settings:
    def __init__(self):
        load_dotenv()
        self.output_path = "./docs/"
        self.chunk_max_chars = 7000
        self.chunk_combine_chars = 1000
        self.chunk_new_chars = 3000
        
    @property
    def openai_api_key(self):
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def groq_api_key(self):
        return os.getenv("GROQ_API_KEY")
