# PDF Ingestion and Processing Pipeline (Without LLM)

## Overview

This project implements a robust pipeline for ingesting, parsing, and processing PDF documents. It is designed to handle documents of varying lengths, extract meaningful summaries and keywords, and store metadata and results in a MongoDB database. The pipeline supports concurrent processing to manage system resources and efficiently handle high volumes of documents.

## Features

- The pipeline ingests PDFs from a specified folder, processes them concurrently to extract summaries, keywords, and metadata, and then stores this information in a MongoDB database, while handling errors and providing performance metrics.

## Requirements

- Python 3.10.5
- MongoDB (local server or may be cloud if have an account on Atlas)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the Repository**:

   ```
   git clone https://github.com/garvpatidar04/garv-patidar-wasserstoff-AiInternTask.git
   cd project_folder
   ```

2. **Set Up a Virtual Environment (optional but recommended)**:
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Required Packages**:
  ```pip install -r requirements.txt```
4. **Set Up MongoDB**:
   Please make sure you have MongoDB installed and running.
    Create a database and collection for storing PDF metadata and processing results.

## Usage
#### Prepare Your PDFs:

* Place the PDF files you want to process in a designated folder on your desktop.                                                  
      Run the Pipeline: `python main.py` 
    
     
