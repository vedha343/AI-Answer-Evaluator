# 📝 AI-Powered Subjective Answer Evaluator

## 📌 Project Overview
This project is an **NLP-based assessment tool** designed to automatically grade subjective (theory) answers. Unlike traditional keyword matching systems, this model utilizes **Semantic Analysis** to understand the *meaning* and context of the student's response.

**Research Motivation:** Manual evaluation of descriptive answers is time-consuming and prone to human bias. This prototype explores the use of **Transformer-based Language Models (BERT)** to provide instant, unbiased, and context-aware grading.

---

## ⚙️ How It Works (The Logic)
The core logic relies on **Vector Space Modeling**:

1.  **Input:** The system accepts a "Reference Answer" (Teacher's Key) and a "Student Answer".
2.  **Vectorization (BERT):** It uses the `sentence-transformers` library (Model: `all-MiniLM-L6-v2`) to convert both text inputs into 384-dimensional dense vector embeddings.
3.  **Similarity Calculation:** It computes the **Cosine Similarity** between the two vectors to determine how close they are in semantic space.
    * *Formula:* $\text{Similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$
4.  **Scoring:** The similarity score (0 to 1) is mapped to a percentage grade.

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Core Logic:** PyTorch, Sentence-Transformers (BERT)
* **Math:** Scikit-Learn (Cosine Similarity)
* **Interface:** Streamlit (Web UI)

---

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/vedha343/AI-Answer-Evaluator.git](https://github.com/vedha343/AI-Answer-Evaluator.git)
    cd AI-Answer-Evaluator
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 🔮 Future Scope
* Integrating **OCR (Optical Character Recognition)** to scan handwritten answer sheets.
* Fine-tuning the BERT model on specific domain datasets (e.g., Medical or Legal texts).
* Adding a "Keyword Check" layer to ensure specific technical terms are present.
