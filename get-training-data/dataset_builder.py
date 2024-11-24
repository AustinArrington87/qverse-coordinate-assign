# usage 
# 2) Using web scraping
# export OPENAI_API_KEY='Enterkey'
#scraper = WebScraper()
#scraper.scrape_category("sports_recreation", "team_sports", "basketball", count=15)

import os
import csv
from pathlib import Path
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import time
import json
import logging
from tqdm import tqdm
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# usage and imports remain the same...

class DatasetBuilder:
    def __init__(self, api_key: str, output_dir: str = "training_data"):
        self.client = OpenAI(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.categories_file = "categories.csv"
        
        # Create base directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir = self.output_dir / "text"
        self.images_dir = self.output_dir / "images"
        self.text_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def load_categories(self) -> List[Dict]:
        """Load categories from CSV file."""
        try:
            categories = []
            csv_path = Path(self.categories_file)
            
            if not csv_path.exists():
                default_categories = [
                    {'category': 'sports_recreation', 'subcategory': 'team_sports'},
                    {'category': 'technology', 'subcategory': 'software'},
                    {'category': 'arts', 'subcategory': 'visual_arts'}
                ]
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['category', 'subcategory'])
                    writer.writeheader()
                    writer.writerows(default_categories)
                
                logger.info(f"Created default categories file at {csv_path}")
                return default_categories
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                categories = [row for row in reader]
            
            logger.info(f"Loaded {len(categories)} categories from {csv_path}")
            return categories
            
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return [{'category': 'general', 'subcategory': 'misc'}]

    def generate_text_prompts(self, category: str, subcategory: str, count: int = 5) -> List[str]:
        """Generate diverse prompts for text content generation."""
        try:
            system_prompt = "Generate simple, clear prompts for content writing. Each prompt should be a single sentence."
            user_prompt = f"Generate {count} different prompts about {subcategory} in {category}. Return only the prompts as a simple JSON array of strings."
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = response.choices[0].message.content
            prompts = json.loads(content)
            
            # Ensure we have strings and the right number of prompts
            prompts = [str(p) for p in prompts[:count]]
            return prompts
            
        except Exception as e:
            logger.error(f"Error generating text prompts: {e}")
            return [f"Write about {subcategory} in the context of {category}."] * count

    def generate_image_prompts(self, category: str, subcategory: str, count: int = 5) -> List[str]:
        """Generate simple, descriptive prompts for image generation."""
        try:
            system_prompt = "Create simple, visual descriptions for image generation. Each prompt should be a clear, single sentence describing a scene or concept."
            user_prompt = f"Generate {count} different visual descriptions about {subcategory} in {category}. Return only the descriptions as a simple JSON array of strings."
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = response.choices[0].message.content
            prompts = json.loads(content)
            
            # Ensure we have strings and the right number of prompts
            prompts = [str(p) for p in prompts[:count]]
            return prompts
            
        except Exception as e:
            logger.error(f"Error generating image prompts: {e}")
            return [f"A simple visual representation of {subcategory} in {category}."] * count

    def generate_text_samples(self, category: str, subcategory: str, 
                            sub_subcategory: str = None, count: int = 10):
        """Generate text samples using GPT-4."""
        output_dir = self.text_dir / category / subcategory
        output_dir.mkdir(parents=True, exist_ok=True)

        prompts = self.generate_text_prompts(category, subcategory, count)
        
        for i, prompt in enumerate(prompts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Generate detailed, realistic content about the given topic."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                content = response.choices[0].message.content
                filename = f"{i+1:03d}_{subcategory.lower()}.txt"
                
                with open(output_dir / filename, 'w') as f:
                    f.write(content)
                
                logger.info(f"Generated text sample {i+1} for {category}/{subcategory}")
                
            except Exception as e:
                logger.error(f"Error generating text sample: {e}")

    def generate_image_samples(self, category: str, subcategory: str, 
                             sub_subcategory: str = None, count: int = 10):
        """Generate image samples using DALL-E."""
        output_dir = self.images_dir / category / subcategory
        output_dir.mkdir(parents=True, exist_ok=True)

        prompts = self.generate_image_prompts(category, subcategory, count)
        
        for i, prompt in enumerate(prompts):
            try:
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                
                image_url = response.data[0].url
                image_response = requests.get(image_url)
                image = Image.open(BytesIO(image_response.content))
                
                filename = f"{i+1:03d}_{subcategory.lower()}.jpg"
                image.save(output_dir / filename)
                
                logger.info(f"Generated image sample {i+1} for {category}/{subcategory}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error generating image sample: {e}")

    def build_dataset(self, sample_count: int = 15):
        """Build complete dataset for all categories."""
        categories = self.load_categories()
        
        for category_data in tqdm(categories, desc="Processing categories"):
            category = category_data['category']
            subcategory = category_data['subcategory']
            
            logger.info(f"\nProcessing {category}/{subcategory}")
            
            # Generate text samples
            self.generate_text_samples(
                category, subcategory, 
                category_data.get('sub_subcategory'),
                count=sample_count
            )
            
            # Generate image samples
            self.generate_image_samples(
                category, subcategory,
                category_data.get('sub_subcategory'),
                count=sample_count
            )
            
            # Rate limiting
            time.sleep(2)

if __name__ == "__main__":
    # Load API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    builder = DatasetBuilder(api_key)
    builder.build_dataset(sample_count=5)  # Start with a smaller sample count for testing
