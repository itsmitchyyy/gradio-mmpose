from typing import Optional
from functools import partial
import gradio as gr
from process_video import VideoProcessor 

def check_video(uploaded_video: Optional[str] = None):
    video_processor = VideoProcessor()
    return video_processor.process_video(uploaded_video)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown('**Input Video**')
            uploaded_video = gr.Video(format='mp4')

    button = gr.Button('Process Video', variant='primary')
    gr.Markdown('**Output Video**')
    output_video = gr.Video()

    button.click(partial(check_video), [uploaded_video], output_video)

gr.close_all()
demo.launch(share=True)

