from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from googlesearch import search


def selenium_web_scrapper(query, num_results=5):
    # Set up Chrome options
    urls = list(search(query, num_results=5))
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    all_text = ""
    import logging
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        logging.info("WebDriver initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize WebDriver: {e}")
        return "Error: Could not initialize WebDriver"
    for url in urls:
        try:
           driver.get(url)
           WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "p"))
            )
           paragraphs = driver.find_elements(By.TAG_NAME, "p")
           page_text = " ".join([p.text.strip() for p in paragraphs if p.text.strip()])
           if page_text:
               all_text += page_text + " "
           #print(f"Scrapped {url}")
        except Exception as e:
            #print(f"Error scraping {url}")
            continue
        if len(all_text) > 1000:
            return all_text.strip()


    return all_text.strip()





