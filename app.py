import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load dataset
df = pd.read_excel("universities.xlsx")

# Display the dataset structure
st.title('University Recommendation System')



# Clean column names
df.columns = df.columns.str.strip()

# Check if 'Budget' column exists
if 'Budget' not in df.columns:
    st.error("Column 'Budget' not found in the dataset.")
else:
    # Handle missing values
    df['Budget'] = df['Budget'].fillna(df['Budget'].mean())
    df['GRE Score'] = df['GRE Score'].fillna(0)
    df['GPA'] = df['GPA'].fillna(df['GPA'].mean())

    # One-hot encode the 'Interest' column
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
    interest_vectors = vectorizer.fit_transform(df['Interest'])
    interest_df = pd.DataFrame(interest_vectors.toarray(), columns=vectorizer.get_feature_names_out())

    # Merge the one-hot encoded interests back into the main dataframe
    df = pd.concat([df.reset_index(drop=True), interest_df.reset_index(drop=True)], axis=1)

    # Define function to gather user input
    def get_user_profile():
        # User input for budget, GRE score, GPA, and interests
        budget = st.number_input("Enter your maximum budget (USD):", min_value=0.0, value=20000.0)
        gre_score = st.number_input("Enter your GRE score:", min_value=0, value=320)
        gpa = st.number_input("Enter your GPA:", min_value=0.0, value=3.5)
        interests = st.text_input("Enter your interests (comma-separated, e.g., AI, Data Science):")

        # Vectorize user-provided interests
        if interests:
            interest_vector = vectorizer.transform([interests]).toarray()
            return budget, gre_score, gpa, interest_vector
        else:
            return None, None, None, None

    # Define function to recommend universities based on user input
    def recommend_universities(budget, gre_score, gpa, interest_vector, top_n=5):
        # Normalize numerical columns
        scaler = MinMaxScaler()
        df[['Budget', 'GRE Score', 'GPA']] = scaler.fit_transform(df[['Budget', 'GRE Score', 'GPA']])

        # Calculate similarity scores
        df['Budget_Similarity'] = 1 - abs(df['Budget'] - budget)
        df['GRE_Similarity'] = 1 - abs(df['GRE Score'] - gre_score)
        df['GPA_Similarity'] = 1 - abs(df['GPA'] - gpa)

        # Calculate interest similarity
        interest_similarity = cosine_similarity(interest_vector, interest_vectors).flatten()

        # Combine all similarity scores
        df['Final_Score'] = (
            0.4 * df['Budget_Similarity'] +
            0.3 * df['GRE_Similarity'] +
            0.2 * df['GPA_Similarity'] +
            0.1 * interest_similarity
        )

        # Sort universities by the final score
        recommendations = df.sort_values(by='Final_Score', ascending=False).head(top_n)

        return recommendations[['University', 'Description', 'Budget', 'GRE Score', 'GPA', 'Interest']]

    # Main UI for user interaction (in the main content area, not sidebar)
    budget, gre_score, gpa, interest_vector = get_user_profile()

    # Button to trigger the recommendation process
    if st.button('Get Recommendations'):
        if budget is not None and gre_score is not None and gpa is not None and interest_vector is not None:
            # Get recommendations based on user input
            recommendations = recommend_universities(budget, gre_score, gpa, interest_vector)

            # Display recommendations
            st.write("### Recommended Universities:")
            st.table(recommendations)
        else:
            st.write("Please enter all the necessary details.")
