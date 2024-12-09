#Import dependencies 
import streamlit as st 
import numpy as np 
import torch
import os 
import imageio
from data_preprocessing import load_video, decode, load_annot
from data_preprocessing import ctc_decode
from model import LipNet
from torchnlp.encoders import LabelEncoder


device = torch.device("cpu")
#Layout config
st.set_page_config(layout= "wide")
st.title("Lip Reader App")
if "loss" not in st.session_state:
    st.session_state["loss"] = False
#Instantiate the sidebar
with st.sidebar:
    st.image("computerVision.jpg")
    
    st.info("""This application is an implemention of the LipNet paper. The goal is to create a computer vision model 
            (LipNet) that can read lips through videos.
            Here is how two versions of the model perform on training and test data. You can find more information in 
            my article on this project on medium. If you want to check the evolution of the loss through epochs, click
            on the button below.
            Hope you enjoy it!!""")
    def click_loss_button():
        st.session_state.loss = True
    def unclick_loss_button():
        st.session_state.loss = False
    col00 , col01 = st.columns(2)

    with col00:
        loss_butt= st.button("Show loss graph", on_click= click_loss_button)
        
    with col01:
        loss_butt= st.button("Hide loss graph", on_click= unclick_loss_button)

    if st.session_state.loss : 
        st.image("../LossGraph.png")

# Save session states
if "split" not in st.session_state:
    st.session_state["split"] = "training_files"
if "model" not in st.session_state:
    st.session_state["selected_model"] = "LIPNET_100_EPOCHS.pt"
if "raw_button" not in st.session_state:
    st.session_state["raw_button"] = False
if "decoded_button" not in st.session_state:
    st.session_state["decoded_button"] = False




# Select videos
files = os.listdir(os.path.join("..", "data", "s1"))
split = ["training files", "validation files"]
st.session_state["split"] = st.selectbox("Choose between training and validation split", split)
n = len(files)
train_files = files[: int(0.9 * n)]
val_files = files[int(0.9 * n) : ]
options = {"training files" : train_files, "validation files" : val_files}
selected_option = st.selectbox(f"choose a video from {st.session_state.split}", options[st.session_state.split] )

col1 , col2 = st.columns(2)




if selected_option : 
    with col1 : 
        st.text("Selected video")
        file_path = os.path.join("..", "data", "s1", selected_option)
        os.system(f"ffmpeg -i {file_path} -vcodec libx264  selected_video.mp4 -y")

        # Renedering video
        video = open("selected_video.mp4", "rb")
        video_bytes = video.read()
        st.video(video_bytes)

        filename = selected_option[ : -4]
        labels_path = os.path.join("..","data", "align\s1", f"{filename}.align")
        labels = load_annot(labels_path, True)
        delim = ""
        st.text(f"The true labels are : {delim.join(decode(labels))}")



    with col2 :
        #select model
        st.info("LIPNET_100_EPOCHS performs the best on test data")
        models = os.listdir(os.path.join("..", "models"))
        st.session_state.selected_model = st.selectbox("Choose a model", models)
        st.info("This is what the ML model sees")
        video = load_video(file_path, from_path=True)
        
        #To GIF 
        frames_np = video.numpy()
        frames_scaled = ((frames_np - frames_np.min()) / (frames_np.max() - frames_np.min()) * 255).astype(np.uint8)
        frames_list = [frame for frame in frames_scaled]
        imageio.mimsave("animation.gif", frames_list, fps = 10)
        st.image("animation.gif", width = 400)

        #Preprocess frames
        frames  = video.float()
        frames = frames.unsqueeze(0)
        frames = frames.permute(1, 2, 3, 0)
        frames = frames.unsqueeze(0)
        #Load model 
        st.info("This is the output of the model")
        model = torch.load(f"../models/{st.session_state.selected_model}", weights_only= False).to(device)
        y_hat = model(frames)
        y_pred = torch.argmax(y_hat, axis = 2)
        y_pred = y_pred.squeeze(dim = 0)
        #st.text(decode(y_pred))
        def click_raw_button():
            st.session_state.raw_button = True
        def unclick_raw_button():
            st.session_state.raw_button = False
        col3 , col4 = st.columns(2)

        with col3:
            raw_butt= st.button("Show raw model outputs", on_click= click_raw_button)
        
        with col4:
            raw_butt= st.button("Hide raw model outputs", on_click= unclick_raw_button)
        
        if st.session_state.raw_button:
            st.text(y_pred)
        

        #Decode output
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        encoder = LabelEncoder(vocab, reserved_labels=['...'], unknown_index=-1)
        
        # Inspect the individual tensors being passed
        

        st.info("This is the decoded output ")
        
        
        
        y_pred_ctc = ctc_decode(y_pred)
        delim = ""
        seq = delim.join(decode(y_pred_ctc))

        col5 , col6 = st.columns(2)
        def click_decoded_button():
            st.session_state.decoded_button = True
        def unclick_decoded_button():
            st.session_state.decoded_button = False
            

        with col5:
            decoded_butt = st.button("Show decoded model output", on_click= click_decoded_button)
        with col6:
            decoded_butt = st.button("Hide decoded model output", on_click= unclick_decoded_button)

        if st.session_state.decoded_button :
            st.text(seq)

