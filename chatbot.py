import streamlit as st
from streamlit_chat import message
from neural_db_loader import generate_queries_chatgpt, search_neural_db, reciprocal_rank_fusion, generate_answers

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

st.title("Insurance Bot")

if len(st.session_state['requests']) == 0 and len(st.session_state['responses']) == 0:
    st.session_state['responses'].append("How can I assist you with your insurance queries?")

# Accept user input
user_input = st.chat_input("Do you have any questions?")

if user_input:
    # Append the user input to the requests
    st.session_state['requests'].append(user_input)

    query_list = generate_queries_chatgpt(user_input)

    reference_list = []
    for query in query_list:
        search_results = search_neural_db(query)
        reference_list.append(search_results)

    r = reciprocal_rank_fusion(dict(zip(query_list, reference_list)))

    ranked_reference_list = [i for i in r.keys()]

    ans = generate_answers(user_input, ranked_reference_list)

    st.session_state['responses'].append(ans)

# Display the conversation history like a chat interface
for i in range(len(st.session_state['responses'])):
    message(st.session_state['responses'][i], key=str(i) + '_bot')
    
    if i < len(st.session_state['requests']):
        message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
