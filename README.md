# AI-Driven Chat Classification System for Drug-Related Messages

## Project Overview

This project implements an AI-driven chat classification system designed to flag drug-related messages in chat conversations. The solution utilizes Natural Language Processing (NLP) techniques and machine learning algorithms to ensure accurate detection while maintaining user privacy by executing the analysis locally on user devices.

## Features

- **High Accuracy:** Achieved 90% classification accuracy using a Random Forest model.
- **Privacy Protection:** Runs locally on user devices, ensuring the security of personal data.
- **Web Scraping:** Utilizes Selenium and Beautiful Soup for efficient data extraction from chat interfaces.
- **NLP Preprocessing:** Incorporates slang replacement and text normalization to enhance model performance.
- **Smart Contract Integration:** Generates CSV files containing flagged content for potential integration with smart contracts on the Ethereum blockchain.
- **Social Network Analysis:** Aims to identify potential perpetrators linked to confirmed cases using social network analysis techniques.

## Technologies Used

- Python
- Libraries: 
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `selenium`
  - `beautifulsoup4`
  - `pickle`
- Machine Learning: 
  - Random Forest
  - BERT (optional for advanced implementation)
- Blockchain: 
  - Ethereum (for smart contract integration)

## Dataset

The project combines a subset of data from Kaggle with a synthesized dataset to enhance model training. The dataset undergoes NLP techniques, including slang word replacements from a predefined dictionary.

## Installation

Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
