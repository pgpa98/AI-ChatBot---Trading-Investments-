import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import os

st.title('Welcome to Tradot!📈 \n Ask me any question on trading/investment')
OPENAI_API_KEY = st.text_input('Enter your Open AI API KEY')
DESTINATION_DIR= st.text_input('Enter your destination file dir')
if st.button("Set API KEY"): 
    st.write("Your OPEN AI API KEY IS" , OPENAI_API_KEY)
    os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

csv_files_uploaded = st.file_uploader(label="Upload your relavent CSV files here", accept_multiple_files=True)

if csv_files_uploaded is not None and len(csv_files_uploaded) == 2:
    for i, csv_file in enumerate(csv_files_uploaded):
        st.write(f"CSV {i+1}: {csv_file.name} uploaded")

    loaders = []
    for csv_file in csv_files_uploaded:
        save_folder = DESTINATION_DIR
        save_path = Path(save_folder, csv_file.name)
        with open(save_path, mode='wb') as w:
            w.write(csv_file.getvalue())

        if save_path.exists():
            st.success(f'File {csv_file.name} is successfully saved!')
            
        loader = CSVLoader(file_path=os.path.join(DESTINATION_DIR, csv_file.name))
        loaders.append(loader)

    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders(loaders)

    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    # Displaying from which file the data was loaded
    st.write(f"Data loaded from CSV 1: {csv_files_uploaded[0].name}")
    st.write(f"Data loaded from CSV 2: {csv_files_uploaded[1].name}")

    # Chatbot interface
    st.title("Chat with your CSV Data")

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def generate_response(user_query):
        response = chain({"question": user_query})
        return response['result']
    
    # We will get the user's input by calling the get_text function
    def get_text():
        input_text = st.text_input("You: ", key="input")
        return input_text
    
    user_input = get_text()

    if user_input:
        output = generate_response(user_input)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

        if len(st.session_state['generated']) == 1:
            st.write(f"Data loaded from CSV 1")
        elif len(st.session_state['generated']) == 2:
            st.write(f"Data loaded from CSV 2")
        else:
            st.write(f"Data loaded from multiple CSVs")
else:
    st.warning("Please upload exactly two CSV files.")
