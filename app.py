import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# function to get response from llama 2 model

def getLlamaResponse(input_text, num_words, blog_style):

    # llama2 model
    llm = CTransformers(model='models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    # prompt template
    template = """
    Write a blog for {blog_style} job profile for a topic {input_text} within {num_words} words. 
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_words"],
                            template=template)
    
    # generate response from llama2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, num_words=num_words))

    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                   page_icon="🤖",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# creating two more columns for additional fields

col1, col2 = st.columns([5,5])

with col1:
    num_words = st.text_input('Num of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', 
                              ('Researcher', 'Data Scientist', 'Regular Person'), 
                              index=0)
    
submit = st.button("Generate")

# final response

if submit:
    st.write(getLlamaResponse(input_text, num_words, blog_style))