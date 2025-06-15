# Project Repository Overview

This repository contains a comprehensive system for web scraping, data processing, and chatbot integration, leveraging multiple tools and techniques to gather and utilize data effectively.

## Scrappers

This section describes the three web scraping scripts included in the repository, each designed for specific use cases:

- **google_web_scrapper_bs4.py**: Utilizes BeautifulSoup4 (BS4) to scrape web content from Google search results. It parses HTML content efficiently, extracting relevant data such as text, links, or metadata from search result pages. Ideal for lightweight scraping tasks that do not require dynamic page interactions.

- **Selenium_webscrapper.py**: Employs Selenium for scraping dynamic websites that rely heavily on JavaScript. This script automates browser interactions, allowing it to handle complex web pages with dynamic content, such as those requiring clicks, scrolls, or form submissions. Suitable for advanced scraping needs.

- **tavily_data_generator.py**: Integrates with the Tavily API to generate structured data from web searches. It focuses on extracting clean, high-quality data for specific queries, making it useful for applications requiring summarized or curated information from the web.

## Main Script (main.py)

The `main.py` file serves as the core of the project, integrating various components to create a cohesive system. It includes the following functionalities:

- **PDF Extraction**: Functions to extract text and data from PDF files, enabling the system to process structured documents like `countries.pdf`.
- **RAG Pipeline Setup**: Configures a Retrieval-Augmented Generation (RAG) pipeline to combine retrieved data with language model outputs for enhanced responses.
- **LLM Setup**: Initializes a Large Language Model (LLM) to generate human-like text for chatbot interactions and content creation.
- **Prompt Template Creation**: Generates a prompt template to train the llm in a certain manner 
- **Web Scraping Decision Algorithm**: Determines when to trigger web scraping based on predefined conditions, optimizing data collection efficiency.
- **Web Scraper Switching**: Allows dynamic switching between different scraping methods (e.g., BS4, Selenium, or Tavily) based on task requirements.
- **run_chatbot**: Integrates all components to run an interactive chatbot that leverages scraped data, PDF content, and the RAG pipeline to provide intelligent responses.

## Demo Notebook (demo.ipynb)

The `demo.ipynb` Jupyter Notebook provides a comprehensive demonstration of the project. It showcases:

- The usage of different scrapers (`google_web_scrapper_bs4.py`, `Selenium_webscrapper.py`, and `tavily_data_generator.py`) with example outputs.
- Various prompt examples to illustrate how the system processes inputs and generates responses.
- The complete workflow, from data extraction (e.g., from `countries.pdf`, which contains basic information about 10 countries) to chatbot interactions.
- Displays the execution of the RAG pipeline, LLM outputs, and promotional template generation.

The `countries.pdf` file serves as a sample dataset, containing basic information about 10 countries, used in the demo to showcase PDF extraction and data utilization.

## Key Features

- **Multiple Scraping Methods**: Supports diverse scraping techniques (BeautifulSoup4, Selenium, and Tavily API) for flexibility in handling static and dynamic web content.
- **PDF Data Extraction**: Extracts and processes data from PDFs, such as the provided `countries.pdf`, for use in downstream tasks.
- **RAG Pipeline Integration**: Combines retrieval and generation for context-aware, accurate chatbot responses.
- **Dynamic Scraper Selection**: Automatically switches between scraping methods based on task needs, improving efficiency and adaptability.
- **LLM-Powered Chatbot**: Runs an intelligent chatbot that integrates scraped web data, PDF content, and generated templates.
- **Promotional Content Generation**: Creates customizable promotional templates from processed data.
- **Comprehensive Demo**: Includes a Jupyter Notebook (`demo.ipynb`) that demonstrates the entire workflow with real examples.
- **Sample Dataset**: Provides `countries.pdf` with information on 10 countries for testing and demonstration purposes.
- **Modular Design**: Organized code structure with reusable components for easy maintenance and scalability.