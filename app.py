import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Free AI Chatbot", page_icon="🤖")

st.title("🤖 Free AI Chatbot — Zero Budget")
st.write("Simple chatbot using a small pretrained model. Works best for short fun chats, jokes, shayari.")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

model = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.form(key='chat_form'):
    user_input = st.text_input('You:', '')
    submit = st.form_submit_button('Send')

if submit and user_input:
    st.session_state.history.append(("You", user_input))
    with st.spinner('AI soch raha hai...'):
        prompt = f"User: {user_input}\nAssistant:" 
        out = model(prompt, max_length=80, num_return_sequences=1)
        generated = out[0]['generated_text']
        if generated.startswith(prompt):
            reply = generated[len(prompt):].strip()
        else:
            reply = generated.strip()
        if len(reply) > 400:
            reply = reply[:400].rsplit('.', 1)[0] + '.'
        st.session_state.history.append(("AI", reply))

for speaker, text in st.session_state.history[::-1]:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**AI:** {text}")

st.write('---')
st.info('Tips: Keep inputs short. For better results you can later switch to a hosted HF inference API or a fine-tuned small model.')
      
