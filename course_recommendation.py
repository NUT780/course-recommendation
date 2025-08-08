import streamlit as st

st.title("Course Recommendation app")
st.markdown("""
### What is this app?

This is a **Course Recommendation System** built with Streamlit.  
Select any course you’re interested in, and the app will suggest similar courses for you.  
It uses machine learning to find the best matches, helping you easily discover new learning opportunities!
""")
try:
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    st.write("")
    df2 = pd.read_pickle("df2.pkl")
    features = pd.read_pickle("features.pkl")
    course_names = pd.read_pickle("course_names.pkl")
    st.write("Data loaded successfully!")

    st.subheader("Available Courses Data")
    st.dataframe(df2)
    course = st.selectbox("Select a course", course_names)
    n = st.slider("Number of recommendations", 1, 10, 3)

    if st.button("Recommend"):
        idx = course_names[course_names == course].index[0]
        course_vector = features.iloc[[idx]]
        model = NearestNeighbors(n_neighbors=n+1, metric='cosine')
        model.fit(features)
        distances, indices = model.kneighbors(course_vector, n_neighbors=n+1)
        recommended_courses = course_names.iloc[indices[0][1:]].values.tolist()
        st.write("Recommended Courses:", recommended_courses)
except Exception as e:
    st.error(f"Error: {e}")