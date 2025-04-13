import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

# Load our knowledge base
def load_knowledge_base():
    # In a real implementation, this would load from a proper database or API 
    # For this prototype, we'll use a simple JSON file with CLAT FAQ data
    kb = {
        "faq": [
            {
                "question": "What is the syllabus for CLAT 2025?",
                "answer": "The CLAT 2025 syllabus includes five sections: English Language, Current Affairs (including General Knowledge), Legal Reasoning, Logical Reasoning, and Quantitative Techniques. Each section tests specific skills relevant to law studies."
            },
            {
                "question": "How many questions are there in the English section?",
                "answer": "The English section of CLAT typically contains 28-32 questions, focusing on comprehension passages, grammar, vocabulary, and verbal reasoning skills."
            },
            {
                "question": "What is the total number of questions in CLAT?",
                "answer": "CLAT exam consists of 150 multiple-choice questions (MCQs) to be completed in 2 hours."
            },
            {
                "question": "Give me last year's cut-off for NLSIU Bangalore.",
                "answer": "The cut-off for National Law School of India University (NLSIU) Bangalore for the previous year was approximately 108-110 out of 150 marks. This varies each year based on difficulty level and applicant performance."
            },
            {
                "question": "What is the CLAT exam pattern?",
                "answer": "CLAT is a 2-hour test with 150 multiple-choice questions across 5 sections: English, Current Affairs, Legal Reasoning, Logical Reasoning, and Quantitative Techniques. Each question carries 1 mark with a negative marking of 0.25 for incorrect answers."
            },
            {
                "question": "How should I prepare for CLAT?",
                "answer": "Effective CLAT preparation includes: 1) Understanding the syllabus thoroughly, 2) Regular reading of newspapers and magazines, 3) Practicing previous years' papers, 4) Taking mock tests, 5) Focusing on improving reading speed, and 6) Developing analytical skills through logical and legal reasoning practice."
            },
            {
                "question": "Which are the top NLUs in India?",
                "answer": "The top National Law Universities (NLUs) in India include NLSIU Bangalore, NALSAR Hyderabad, WBNUJS Kolkata, NLU Delhi, and NLU Jodhpur, though rankings may vary year to year based on various factors."
            },
            {
                "question": "What is the eligibility criteria for CLAT?",
                "answer": "For CLAT undergraduate programs (B.A. LL.B), candidates must have completed 10+2 or equivalent with a minimum of 45% marks (40% for SC/ST categories). For CLAT postgraduate programs (LL.M), candidates need a LL.B degree or equivalent with 55% marks (50% for SC/ST)."
            },
            {
                "question": "When will CLAT 2025 be conducted?",
                "answer": "CLAT 2025 is expected to be conducted in May 2025, though the exact date will be announced by the Consortium of NLUs. Candidates should regularly check the official CLAT website for updates."
            },
            {
                "question": "How is the CLAT score calculated?",
                "answer": "CLAT score is calculated by awarding 1 mark for each correct answer and deducting 0.25 marks for each incorrect answer. No marks are deducted for unattempted questions. The final score is the sum of marks obtained across all sections."
            },
            {
                "question": "What books are recommended for CLAT preparation?",
                "answer": "Recommended books for CLAT include: Universal's Lexis Nexis Guide for CLAT, Legal Aptitude for CLAT by A.P. Bhardwaj, Word Power Made Easy by Norman Lewis for vocabulary, and R.S. Aggarwal for logical reasoning and quantitative techniques. Additionally, regular reading of newspapers like The Hindu and Indian Express is advised."
            },
            {
                "question": "What is CLAT?",
                "answer": "CLAT (Common Law Admission Test) is a centralized national level entrance test for admissions to 22+ National Law Universities (NLUs) in India. It's conducted for both undergraduate (UG) and postgraduate (PG) law programs. CLAT was first conducted in 2008 to standardize the law admission process across the country."
            },
            {
                "question": "How many attempts are allowed for CLAT?",
                "answer": "There is no limit on the number of attempts a candidate can make for CLAT. As long as you meet the eligibility criteria, you can appear for the exam any number of times to improve your score."
            },
            {
                "question": "What is the application fee for CLAT 2025?",
                "answer": "The application fee for CLAT 2025 is expected to be around Rs. 4,000 for General/OBC/PWD categories and Rs. 3,500 for SC/ST/BPL categories. The exact fee may be updated on the official CLAT website."
            },
            {
                "question": "What is the age limit for CLAT?",
                "answer": "There is no specific upper age limit for CLAT. However, candidates must have completed 17 years of age to be eligible for admission to the undergraduate program."
            },
            {
                "question": "How to register for CLAT 2025?",
                "answer": "To register for CLAT 2025: 1) Visit the official CLAT website (consortiumofnlus.ac.in), 2) Create an account, 3) Fill in the application form with personal and academic details, 4) Upload required documents (photograph, signature, category certificate if applicable), 5) Pay the application fee online, 6) Submit the form and download the confirmation page."
            },
            {
                "question": "What is the marking scheme for CLAT?",
                "answer": "CLAT follows a simple marking scheme: +1 mark for each correct answer and -0.25 mark for each incorrect answer. There is no negative marking for unattempted questions. The total raw score is calculated based on this formula, and then ranks are assigned based on the total marks obtained."
            },
            {
                "question": "How can I improve my legal reasoning section score?",
                "answer": "To improve in the Legal Reasoning section: 1) Develop a clear understanding of legal principles, 2) Read judgments and legal articles to familiarize yourself with legal language, 3) Practice with previous years' papers, 4) Work on your analytical skills, 5) Take timed mock tests, 6) Learn to identify the principle and facts from a legal passage, and 7) Study basic legal maxims and concepts."
            },
            {
                "question": "What is the current affairs section in CLAT?",
                "answer": "The Current Affairs section in CLAT tests knowledge of recent events and general awareness. It covers national and international news, important personalities, awards, sports, books, authors, and basic knowledge of the Indian constitution and legal system. Unlike traditional GK, CLAT focuses on analyzing current events rather than mere factual recall."
            },
            {
                "question": "How important are mock tests for CLAT preparation?",
                "answer": "Mock tests are extremely important for CLAT preparation because they: 1) Familiarize you with the actual exam pattern and interface, 2) Help improve time management, 3) Identify your strengths and weaknesses, 4) Build exam temperament and reduce anxiety, 5) Allow you to practice different strategies, and 6) Give you a realistic assessment of your preparation level and expected rank."
            },
            {
                "question": "What is the difficulty level of CLAT?",
                "answer": "CLAT is considered moderately difficult. The exam doesn't test extremely advanced concepts but requires good analytical skills, reading comprehension abilities, and time management. The competition is intense due to limited seats at premier NLUs. The difficulty level varies slightly each year, with recent trends showing more emphasis on comprehension-based questions across all sections."
            },
            {
                "question": "How many seats are available through CLAT?",
                "answer": "Approximately 2,500-3,000 seats are available across all participating NLUs through CLAT for undergraduate programs. The exact number varies each year as new NLUs join the consortium. The premier NLUs like NLSIU Bangalore, NALSAR Hyderabad, and NLU Delhi typically offer 120-180 seats each."
            },
            {
                "question": "What is the reservation policy for CLAT admissions?",
                "answer": "Each NLU has its own reservation policy based on state and central government norms. Generally, there is reservation for SC/ST/OBC categories (approximately 49.5%), persons with disabilities (5%), and in some cases, state domicile quotas. Some NLUs have specific quotas for foreign nationals, Kashmir migrants, or defense personnel wards. Check the specific university's admission policy for detailed information."
            },
            {
                "question": "How should I prepare for the logical reasoning section?",
                "answer": "For Logical Reasoning preparation: 1) Master basic concepts like syllogisms, assumptions, inferences, and arguments, 2) Practice puzzles, seating arrangements, blood relations, and series problems, 3) Develop critical thinking skills, 4) Take timed practice tests to improve speed, 5) Learn to identify patterns and relationships, 6) Use books like R.S. Aggarwal's Logical Reasoning or dedicated CLAT preparation material."
            },
            {
                "question": "What are the important dates for CLAT 2025?",
                "answer": "Important dates for CLAT 2025 (tentative): Application form release - August/September 2024; Last date for application - March/April 2025; Admit card release - 2 weeks before exam; Exam date - May 2025; Result declaration - Within 2-3 weeks of exam. For exact dates, monitor the official CLAT website regularly."
            },
            {
                "question": "How can I get scholarships for studying at NLUs?",
                "answer": "Scholarships at NLUs include: 1) Merit-based scholarships offered by individual NLUs, 2) Government scholarships like Post-Matric Scholarship for SC/ST students, 3) IDIA (Increasing Diversity by Increasing Access) scholarships, 4) Corporate scholarships from law firms, 5) Need-based fee waivers, and 6) State government scholarships. Check each NLU's website for specific scholarship details and eligibility criteria."
            },
            {
                "question": "What is the scope after completing a law degree from an NLU?",
                "answer": "After graduating from an NLU, opportunities include: 1) Joining law firms (corporate, litigation, IPR, etc.), 2) Independent practice as an advocate, 3) Legal roles in corporations, 4) Judicial services, 5) Civil services, 6) Academic careers, 7) Legal journalism, 8) NGO and human rights work, 9) International organizations like UN, WTO, etc., 10) Further specialization through LL.M. in India or abroad, and 11) Legal research and policy-making roles."
            },
            {
                "question": "What is the English section format in CLAT?",
                "answer": "The English section in CLAT primarily tests reading comprehension skills through passages followed by questions. It evaluates: 1) Understanding of complex texts, 2) Vocabulary in context, 3) Inference and analysis abilities, 4) Grammar and language usage, 5) Summarizing and drawing conclusions. The focus is on comprehension rather than standalone grammar or vocabulary questions."
            },
            {
                "question": "How to manage time during the CLAT exam?",
                "answer": "Effective time management strategies for CLAT: 1) Allocate time to each section based on your strengths (approximately 35-40 minutes for English, 25-30 minutes for Legal Reasoning, and the remaining time divided among other sections), 2) Attempt easy questions first, 3) Don't spend more than 1-1.5 minutes on any single question, 4) Skip difficult questions and return to them later, 5) Practice with timed mock tests, and 6) Keep 5-10 minutes at the end for review."
            },
            {
                "question": "Which NLUs accept CLAT scores?",
                "answer": "NLUs accepting CLAT scores include: NLSIU Bangalore, NALSAR Hyderabad, NLIU Bhopal, WBNUJS Kolkata, NLU Jodhpur, HNLU Raipur, GNLU Gandhinagar, RMLNLU Lucknow, RGNUL Patiala, CNLU Patna, NUALS Kochi, NUSRL Ranchi, NLUO Cuttack, MNLU Mumbai, MNLU Nagpur, MNLU Aurangabad, TNNLS Tiruchirappalli, DNLU Jabalpur, NLUJA Assam, HPNLU Shimla, MNLU Delhi, and Dr. BRAOU Lucknow. Additional law schools may also accept CLAT scores for admissions."
            },
            {
                "question": "What is the quantitative techniques section in CLAT?",
                "answer": "The Quantitative Techniques section in CLAT tests basic mathematical concepts and numerical ability. It covers: 1) Elementary mathematics (up to 10th grade level), 2) Data interpretation (graphs, charts, tables), 3) Logical reasoning with numbers, 4) Basic arithmetic operations, 5) Percentages, ratios, and averages, 6) Profit and loss, 7) Simple and compound interest, and 8) Speed, time, and distance problems. The focus is on application rather than theoretical knowledge."
            },
            {
                "question": "How different is CLAT from AILET?",
                "answer": "CLAT and AILET (All India Law Entrance Test for NLU Delhi) differ in several ways: 1) CLAT provides admission to 22+ NLUs while AILET is specifically for NLU Delhi, 2) CLAT has 150 questions in 2 hours while AILET has 150 questions in 1.5 hours, 3) AILET typically includes a separate GK section while CLAT combines it with Current Affairs, 4) AILET may include subjective questions in some years, 5) The application processes are separate, and 6) The exams are held on different dates, allowing candidates to appear for both."
            },
            {
                "question": "What are the best coaching institutes for CLAT?",
                "answer": "Popular coaching institutes for CLAT preparation include: Career Launcher (CL), Legal Edge, IMS, Law Prep Tutorial, Sriram Law Academy, LST, Legitimate, Raos IAS, and Clat Possible. Many offer both offline and online coaching programs. However, self-study with good materials and consistent practice is equally effective for many successful candidates. The choice of coaching should depend on your learning style, financial capacity, and geographic location."
            },
            {
                "question": "Is CLAT conducted online or offline?",
                "answer": "CLAT is conducted in online mode (computer-based test) at designated test centers across various cities in India. Candidates receive a computer terminal to take the test, where they can navigate through questions, mark answers, and review their responses before final submission. The online format allows for real-time clock monitoring and immediate calculation of results."
            },
            {
                "question": "How do I prepare for the English section of CLAT?",
                "answer": "To prepare for the English section: 1) Read diverse materials including newspapers, magazines, novels, and academic articles to improve comprehension, 2) Practice reading passages and answering questions within time limits, 3) Build vocabulary through contextual learning, 4) Work on grammar fundamentals, 5) Learn to identify main ideas, arguments, and inferences in complex texts, 6) Practice summarization skills, and 7) Take mock tests and analyze mistakes to improve accuracy."
            },
            {
                "question": "What is the counseling process after CLAT results?",
                "answer": "The CLAT counseling process involves: 1) Registration on the counseling portal, 2) Payment of counseling fee, 3) Filling preferences for NLUs in order of priority, 4) Seat allotment based on rank and category, 5) Accepting the allotted seat by paying the required fee, 6) Multiple rounds of counseling to fill vacant seats, and 7) Reporting to the allotted NLU for document verification and admission. The entire process is typically online, followed by physical reporting to the allotted university."
            },
            {
                "question": "What is the fee structure at top NLUs?",
                "answer": "Fee structure at top NLUs varies, but typically ranges between Rs. 2-3 lakhs per year for undergraduate programs. NLSIU Bangalore and NLU Delhi tend to have higher fees (approximately Rs. 2.5-3.5 lakhs per year), while some state NLUs may have slightly lower fees. This generally includes tuition, hostel accommodation, and basic amenities. Additional expenses for books, extracurricular activities, and personal expenses should also be budgeted for."
            },
            {
                "question": "How to stay updated with current affairs for CLAT?",
                "answer": "To stay updated with Current Affairs for CLAT: 1) Read quality newspapers like The Hindu, Indian Express, or Livemint daily, 2) Follow dedicated CLAT current affairs compilations, 3) Use apps like Inshorts or daily current affairs websites, 4) Watch news analysis programs, 5) Create notes on important events with legal or constitutional implications, 6) Follow Supreme Court judgments and legal developments, 7) Join study groups to discuss current events, and 8) Read monthly magazines like Pratiyogita Darpan or Competition Success Review."
            },
            {
                "question": "What are the best online resources for CLAT preparation?",
                "answer": "Valuable online resources for CLAT preparation include: 1) Official CLAT website for sample papers, 2) Online learning platforms like Unacademy and CLATapult, 3) Legal news websites such as Live Law and Bar and Bench, 4) YouTube channels dedicated to CLAT preparation, 5) Mock test series from Career Launcher or Legal Edge, 6) Current affairs compilations from CLATGyan or Law School 101, 7) Online forums and discussion groups, and 8) Legal databases like SCC Online or Manupatra for understanding legal concepts."
            }
        ]
    }
    return kb

# Process the knowledge base for text similarity searching
def preprocess_knowledge_base(kb):
    questions = [item["question"] for item in kb["faq"]]
    answers = [item["answer"] for item in kb["faq"]]
    
    #  TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    
    return vectorizer, question_vectors, questions, answers

# Find the most similar question in our knowledge base
def find_best_answer(query, vectorizer, question_vectors, questions, answers, similarity_threshold=0.3):
    # Process the query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity scores
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    
    # Find the most similar question
    best_match_idx = np.argmax(similarities)
    max_similarity = similarities[best_match_idx]
    
    # Only return an answer if similarity is above threshold
    if max_similarity >= similarity_threshold:
        return answers[best_match_idx], questions[best_match_idx], max_similarity
    else:
        return None, None, 0.0

# Clean user input
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    st.set_page_config(
        page_title="CLAT Exam Assistant",
        page_icon="⚖️",
        layout="wide"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOxrzVVKPvlLvs_1MCKH8KRDoahlvW0R97kw&s", width=150)
        st.markdown("### NLTI")
        st.markdown("National Law Talent Initiative")
    
    with col2:
        st.title("CLAT Exam Assistant")
        st.markdown("Ask any question about CLAT exam preparation, syllabus, or admission process.")
    
    st.markdown("---")
    
    # Load and preprocess knowledge base
    kb = load_knowledge_base()
    vectorizer, question_vectors, questions, answers = preprocess_knowledge_base(kb)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            is_user = message["role"] == "user"
            
            message_container = st.container()
            with message_container:
                if is_user:
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**CLAT Assistant:** {message['content']}")
                    
                st.markdown("---")
    
    query = st.text_input("Ask a question about CLAT...", key="query_input")
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.button("Submit", use_container_width=True)
    with col2:
        clear_button = st.button("Clear Chat", use_container_width=True)
    
    # Clear chat history when Clear Chat is clicked
    if clear_button:
        st.session_state.messages = []
        st.experimental_rerun()
    
    if submit_button and query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        cleaned_query = clean_text(query)
        
        answer, matched_question, confidence = find_best_answer(
            cleaned_query, vectorizer, question_vectors, questions, answers
        )
        
        if answer:
            response = answer
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        else:
            response = "I'm sorry, I don't have enough information to answer that question accurately. Please try rephrasing your question or ask our human mentors for assistance."
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.experimental_rerun()
    
    if st.session_state.messages and len(st.session_state.messages) >= 2:
        last_user_query = st.session_state.messages[-2]["content"] if st.session_state.messages[-2]["role"] == "user" else None
        
        if last_user_query:
            cleaned_query = clean_text(last_user_query)
            answer, matched_question, confidence = find_best_answer(
                cleaned_query, vectorizer, question_vectors, questions, answers
            )
            
            if answer and confidence > 0.3:
                with st.expander("See matched question details"):
                    st.markdown(f"**Matched Question:** {matched_question}")
                    st.markdown(f"**Confidence Score:** {confidence:.2f}")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This CLAT Exam Assistant helps aspirants get quick answers to their CLAT-related queries.
            
            It uses Natural Language Processing to understand your questions and match them with 
            the most relevant information from our knowledge base.
            
            **Features:**
            - Get information about CLAT syllabus
            - Learn about exam pattern
            - Find out about cutoffs
            - Get preparation tips
            
            For more detailed guidance, consider connecting with our expert mentors.
            """
        )
        
        st.header("Top CLAT Resources")
        st.markdown(
            """
            - **Official CLAT Website**: consortiumofnlus.ac.in
            - **Recommended Books**: 
              - Universal's Guide to CLAT
              - Legal Reasoning by A.P. Bhardwaj
              - Word Power Made Easy
            - **Mock Tests**: Take at least 30 full-length mocks
            - **Daily Practice**: 2-3 hours of focused study
            """
        )
        
        st.header("Sample Questions")
        
        question_categories = {
            "Exam Basics": [
                "What is CLAT?", 
                "What is the syllabus for CLAT 2025?",
                "What is the CLAT exam pattern?"
            ],
            "Preparation": [
                "How should I prepare for CLAT?",
                "What books are recommended for CLAT preparation?",
                "How to manage time during the CLAT exam?"
            ],
            "Universities & Admissions": [
                "Which are the top NLUs in India?",
                "What is the fee structure at top NLUs?",
                "What is the counseling process after CLAT results?"
            ]
        }
        
        for category, questions_list in question_categories.items():
            with st.expander(category):
                for q in questions_list:
                    if st.button(q, key=f"btn_{q[:20]}"):
                        st.session_state.messages.append({"role": "user", "content": q})
                        
                        cleaned_q = clean_text(q)
                        answer, matched_question, confidence = find_best_answer(
                            cleaned_q, vectorizer, question_vectors, questions, answers
                        )
                        
                        if answer:
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I don't have enough information to answer that question accurately."})
                        
                        st.experimental_rerun()

if __name__ == "__main__":
    main()