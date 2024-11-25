# Run either script --> 1) openAI image generation or ...
#builder = DatasetBuilder(api_key)
#builder.build_dataset(sample_count=15)

# 2) Using web scraping
#scraper = WebScraper()
#scraper.scrape_category("sports_recreation", "team_sports", "basketball", count=15)
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from pathlib import Path
import logging
import json
import urllib.parse
import csv
import pandas as pd
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, output_dir: str = "training_data", categories_file: str = "categories.csv"):
        self.output_dir = Path(output_dir)
        self.categories_file = categories_file
        self.setup_driver()
        self.setup_image_urls()
        
    def setup_image_urls(self):
        """Setup diverse image URLs for different categories."""
        self.category_urls = {
            'arts_entertainment': {
                'visual_arts': [
                    'https://images.unsplash.com/photo-1547891654-e66ed7ebb968',  # art gallery
                    'https://images.unsplash.com/photo-1513364776144-60967b0f800f',  # art supplies
                    'https://images.unsplash.com/photo-1571115764595-644a1f56a55c',  # painting
                    'https://images.unsplash.com/photo-1520420097861-e4959843b682',  # sculpture
                    'https://images.unsplash.com/photo-1578022761797-b8636ac1773c'   # photography
                ],
                'music': [
                    'https://images.unsplash.com/photo-1511379938547-c1f69419868d',  # music
                    'https://images.unsplash.com/photo-1460661419201-fd4cecdf8a8b',  # concert
                    'https://images.unsplash.com/photo-1514320291840-2e0a9bf2a9ae',  # musician
                    'https://images.unsplash.com/photo-1465225314224-587cd83d322b',  # piano
                    'https://images.unsplash.com/photo-1486092642310-0c4e84309adb'   # guitar
                ],
                'film_tv': [
                    'https://images.unsplash.com/photo-1485846234645-a62644f84728',  # cinema
                    'https://images.unsplash.com/photo-1524712245354-2c4e5e7121c0',  # tv
                    'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba',  # film
                    'https://images.unsplash.com/photo-1478720568477-152d9b164e26',  # movie
                    'https://images.unsplash.com/photo-1595769816263-9b910be24d5f'   # streaming
                ]
            }
        }
        
        # Add a default set of URLs for any unlisted categories
        self.default_urls = [
            'https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3',
            'https://images.unsplash.com/photo-1497436072909-60f360e1d4b1',
            'https://images.unsplash.com/photo-1493612276216-ee3925520721',
            'https://images.unsplash.com/photo-1531297484001-80022131f5a1',
            'https://images.unsplash.com/photo-1527219525722-f9767a7f2884'
        ]

    def load_categories(self):
        """Load categories from CSV file."""
        try:
            df = pd.read_csv(self.categories_file)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return []

    def setup_driver(self):
        """Setup Chrome driver with additional options."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)

    def get_image_urls(self, category: str, subcategory: str, count: int = 5):
        """Get appropriate image URLs for the category."""
        try:
            # Try to get category-specific URLs
            urls = self.category_urls.get(category, {}).get(subcategory, self.default_urls)
            
            # If we have more URLs than needed, randomly select
            if len(urls) > count:
                return random.sample(urls, count)
            # If we need more URLs, cycle through existing ones
            return (urls * (count // len(urls) + 1))[:count]
            
        except Exception as e:
            logger.error(f"Error getting image URLs: {e}")
            return self.default_urls[:count]

    def scrape_images(self, query: str, category: str, subcategory: str, 
                     sub_subcategory: str = None, count: int = 5):
        """Scrape images."""
        try:
            parts = [self.output_dir, "images", category, subcategory]
            if sub_subcategory:
                parts.append(sub_subcategory)
            output_dir = Path(*parts)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            urls = self.get_image_urls(category, subcategory, count)
            
            for i, url in enumerate(urls[:count]):
                try:
                    url = f"{url}?w=1200&q=80&rand={time.time()}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        filename = f"{i+1:03d}_{subcategory.lower()}"
                        if sub_subcategory:
                            filename += f"_{sub_subcategory.lower()}"
                        filename += ".jpg"
                        
                        with open(output_dir / filename, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Saved image {i+1}/{count}")
                        
                except Exception as e:
                    logger.error(f"Error downloading image {i+1}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in image scraping: {e}")

    def scrape_text(self, query: str, category: str, subcategory: str, 
                    sub_subcategory: str = None, count: int = 5):
        """Scrape text content."""
        try:
            parts = [self.output_dir, "text", category, subcategory]
            if sub_subcategory:
                parts.append(sub_subcategory)
            output_dir = Path(*parts)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            search_url = f"https://www.google.com/search?q={query}"
            self.driver.get(search_url)
            
            links = self.driver.find_elements(By.CSS_SELECTOR, "div.g a")
            processed = 0
            
            for link in links:
                if processed >= count:
                    break
                    
                try:
                    url = link.get_attribute('href')
                    if url and url.startswith('http'):
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            content = ' '.join([p.text for p in soup.find_all('p')])
                            if len(content) > 100:
                                filename = f"{processed+1:03d}_{subcategory.lower()}"
                                if sub_subcategory:
                                    filename += f"_{sub_subcategory.lower()}"
                                filename += ".txt"
                                
                                with open(output_dir / filename, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                logger.info(f"Saved text {processed+1}/{count}")
                                processed += 1
                except Exception as e:
                    logger.error(f"Error scraping text content: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in text scraping: {e}")

    def scrape_category(self, category: str, subcategory: str, 
                       sub_subcategory: str = None, count: int = 5):
        """Scrape both images and text for a category."""
        try:
            query_parts = [category.replace('_', ' '), subcategory.replace('_', ' ')]
            if sub_subcategory:
                query_parts.append(sub_subcategory.replace('_', ' '))
            query = ' '.join(query_parts)
            
            logger.info(f"Processing: {query}")
            
            self.scrape_images(query, category, subcategory, sub_subcategory, count)
            self.scrape_text(query, category, subcategory, sub_subcategory, count)
            
        except Exception as e:
            logger.error(f"Error processing category: {e}")

    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    try:
        scraper = WebScraper()
        logger.info("Web scraper initialized successfully")
        
        categories = scraper.load_categories()
        logger.info(f"Loaded {len(categories)} categories from categories.csv")
        
        for idx, cat in enumerate(categories, 1):
            try:
                logger.info(f"\nProcessing category {idx}/{len(categories)}: "
                           f"{cat['category']}/{cat['subcategory']}" +
                           (f"/{cat['sub_subcategory']}" if pd.notna(cat['sub_subcategory']) else ""))
                
                scraper.scrape_category(
                    cat['category'],
                    cat['subcategory'],
                    cat['sub_subcategory'] if pd.notna(cat['sub_subcategory']) else None,
                    count=5
                )
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing category {idx}: {e}")
                continue
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'scraper' in locals():
            del scraper

if __name__ == "__main__":
    main()
