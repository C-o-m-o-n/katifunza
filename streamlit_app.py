import streamlit as st
from magic.magic import retrieved_docs

st.title("Welcome to Katifunza")
st.write(
    "Where we simplify the constitution for you"
)

txt = st.text_area(
    "Enter your question about the constitution",
    )

if st.button("Send"):
    st.write("The output will appear bellow:")
    ans_list = retrieved_docs(txt)
    for ans in ans_list:
        st.write(ans.page_content)
