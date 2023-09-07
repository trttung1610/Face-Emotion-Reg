import streamlit as st
from process_emotion_video import process_emotion

def main():
    st.title("Emotion Recognition in Video")
    option = st.radio("Choose an option:", ("Upload a video file", "Example Video"))
    if option == "Upload a video file":
        # Upload video file
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    
        if uploaded_file is not None:
            st.write("Processing...")
    
            # Process the video and get the output path
            output_path = process_emotion(uploaded_file)
    
            if output_path.endswith(".mp4"):
                st.video(output_path)
    
                # Add a download button for the processed video
                st.download_button(
                    label="Download Processed Video",
                    data=output_path,
                    key="download_processed_video",
                    file_name="processed_video.mp4",
                )
            else:
                st.error("An error occurred during video processing.")
    else:
        example_video_path = 'demo_video.mp4'
        st.video(example_video_path)
        st.download_button(
            label="Download Processed Video",
            data=example_video_path,
            key="download_demo_video",
            file_name="demo_video.mp4",
            )
    

if __name__ == "__main__":
    main()
