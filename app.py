import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# 1. LOAD THE AI MODEL ( The "Brain" )
# ---------------------------------------------------------
# We use @st.cache_resource so the model loads only once.
# This makes the app run fast after the first load.
@st.cache_resource
def load_model():
    # 'all-MiniLM-L6-v2' is a small, fast BERT model ideal for laptops.
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------------------------------------------------
# 2. THE USER INTERFACE ( The "Face" )
# ---------------------------------------------------------
st.title("📝 AI Subjective Answer Evaluator")
st.markdown("### A Research Prototype using NLP (BERT)")
st.write("This tool grades answers based on **Semantic Meaning**, not just keyword matching.")

# Create two columns for a cleaner look (Optional, but looks professional)
col1, col2 = st.columns(2)

with col1:
    st.info("Teacher's Section")
    teacher_answer = st.text_area("Enter Standard Answer (Key):", height=150, key="teacher")

with col2:
    st.info("Student's Section")
    student_answer = st.text_area("Enter Student's Answer:", height=150, key="student")

# ---------------------------------------------------------
# 3. THE LOGIC ( The "Math" )
# ---------------------------------------------------------
if st.button("Evaluate Answer 🚀"):
    if not teacher_answer or not student_answer:
        st.error("⚠️ Please fill in both the Teacher and Student answer boxes.")
    else:
        with st.spinner("Analyzing semantic similarity..."):
            # Step A: Vectorization
            # Convert text into a list of numbers (Embeddings)
            embedding_1 = model.encode(teacher_answer, convert_to_tensor=True)
            embedding_2 = model.encode(student_answer, convert_to_tensor=True)
            
            # Step B: Cosine Similarity
            # Calculate the cosine of the angle between the two vectors
            score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
            
            # Step C: Convert to Percentage
            percentage = round(score * 100, 2)

        # ---------------------------------------------------------
        # 4. DISPLAY RESULTS
        # ---------------------------------------------------------
        st.divider()
        st.header(f"AI Grade: {percentage}/100")
        
        # Give feedback based on the score
        if percentage >= 85:
            st.success("🌟 Excellent! The meaning matches perfectly.")
        elif percentage >= 60:
            st.warning("⚠️ Good attempt, but missed some key context.")
        else:
            st.error("❌ The answer is incorrect or irrelevant.")
            
        # EXPANDER: Only for the Professor/Viva (Hidden by default)
        with st.expander("See Technical Details (For Viva)"):
            st.write(f"**Cosine Similarity Score:** {score:.4f}")
            st.write("**Model Used:** all-MiniLM-L6-v2 (Pre-trained BERT)")
            st.write("This score represents the semantic closeness of the two vectors.")