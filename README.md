# CLAT Exam Assistant Chatbot

This project is a simple NLP-powered chatbot that answers CLAT-related queries using Streamlit. It provides a user-friendly interface for law aspirants to get quick answers about CLAT exam preparation, syllabus, and admission processes.

## Features

- Interactive chat interface built with Streamlit
- NLP-based query matching using TF-IDF vectorization and cosine similarity
- Knowledge base with common CLAT FAQs
- Confidence scoring for answer relevance
- Sample questions for easy exploration
- Mobile-responsive design

## Technical Architecture

The chatbot uses a simple but effective NLP approach:

1. **Knowledge Base**: A collection of question-answer pairs related to CLAT exams
2. **Text Processing**: Cleaning and normalizing user queries
3. **Vectorization**: Converting text to TF-IDF vectors
4. **Similarity Matching**: Using cosine similarity to find the most relevant answer
5. **Threshold Filtering**: Only providing answers with sufficient confidence

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/saurabh23011/Clat-chatbot.git
cd clat-exam-assistant
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Project Structure

```
clat-exam-assistant/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Scaling to a GPT-based Solution

To scale this prototype to a more advanced GPT-based model:

1. **Fine-tuning**: Use NLTI's proprietary content to fine-tune a GPT model specifically for legal education queries
2. **Expanded Knowledge Base**: Integrate comprehensive CLAT resources, past papers, and expert advice
3. **Personalization**: Add user profiles to customize responses based on preparation level and goals
4. **Multi-modal Support**: Extend to support document parsing for legal cases and study materials
5. **Integration with Mentor System**: Connect with the mentor recommendation system (Task 1) to suggest human mentors for complex queries

This would require:
- Setting up a pipeline for data collection and model fine-tuning
- Implementing a more robust backend with proper database integration
- Adding authentication for personalized experiences
- Deploying to cloud infrastructure for scalability

## Requirements

```
streamlit
pandas
numpy
scikit-learn
```

## Evaluation Metrics

The chatbot performance can be measured by:
- Accuracy of responses
- Response relevance (cosine similarity score)
- User satisfaction (feedback mechanism)
- Coverage of CLAT topics
