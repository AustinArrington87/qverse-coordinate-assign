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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from pathlib import Path
import hashlib
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.setup_driver()
        
    def setup_driver(self):
        """Setup headless Chrome driver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def scrape_images(self, query: str, category: str, subcategory: str, count: int = 10):
        """Scrape images from Google Images."""
        output_dir = self.output_dir / "images" / category / subcategory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        search_url = f"https://www.google.com/search?q={query}&tbm=isch"
        logger.info(f"Scraping images for query: {query}")
        self.driver.get(search_url)
        
        # Scroll to load more images
        for _ in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        images = self.driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
        
        for i, img in enumerate(images[:count]):
            try:
                src = img.get_attribute('src')
                if src and src.startswith('http'):
                    response = requests.get(src)
                    if response.status_code == 200:
                        filename = f"{i+1:03d}_{subcategory.lower()}.jpg"
                        with open(output_dir / filename, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Saved image {i+1}/{count} for {category}/{subcategory}")
            except Exception as e:
                logger.error(f"Error downloading image: {e}")

    def scrape_text(self, query: str, category: str, subcategory: str, count: int = 10):
        """Scrape relevant text content from websites."""
        output_dir = self.output_dir / "text" / category / subcategory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        search_url = f"https://www.google.com/search?q={query}"
        logger.info(f"Scraping text for query: {query}")
        self.driver.get(search_url)
        
        links = self.driver.find_elements(By.CSS_SELECTOR, "div.g a")
        processed = 0
        
        for link in links:
            if processed >= count:
                break
                
            try:
                url = link.get_attribute('href')
                if url and url.startswith('http'):
                    response = requests.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Get main content
                        content = ' '.join([p.text for p in soup.find_all('p')])
                        if len(content) > 100:  # Minimum content length
                            filename = f"{processed+1:03d}_{subcategory.lower()}.txt"
                            with open(output_dir / filename, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logger.info(f"Saved text {processed+1}/{count} for {category}/{subcategory}")
                            processed += 1
            except Exception as e:
                logger.error(f"Error scraping text: {e}")

    def scrape_category(self, category: str, subcategory: str, 
                       sub_subcategory: str = None, count: int = 10):
        """Scrape both images and text for a category."""
        query = f"{category} {subcategory}"
        if sub_subcategory:
            query += f" {sub_subcategory}"
            
        self.scrape_images(query, category, subcategory, count)
        self.scrape_text(query, category, subcategory, count)
        
    def __del__(self):
        """Cleanup method to close the browser when done."""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    # Example categories to scrape
    categories = [
        {
            'category': 'sports_recreation',
            'subcategory': 'basketball',
            'count': 5
        },
        {
            'category': 'technology',
            'subcategory': 'artificial_intelligence',
            'count': 5
        },
        {
            'category': 'arts',
            'subcategory': 'digital_art',
            'count': 5
        }
    ]
    
    try:
        scraper = WebScraper()
        logger.info("Web scraper initialized successfully")
        
        for cat in categories:
            logger.info(f"\nProcessing category: {cat['category']}/{cat['subcategory']}")
            scraper.scrape_category(
                cat['category'],
                cat['subcategory'],
                count=cat['count']
            )
            time.sleep(2)  # Add delay between categories
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'scraper' in locals():
            del scraper  # This will trigger the cleanup

if __name__ == "__main__":
    main()
