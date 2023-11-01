import gradio as gr
 
def capitalize_text(input_text):
    return input_text.upper()
 
iface = gr.Interface(fn=capitalize_text, inputs="text", outputs="text")
iface.launch(share=True, server_name='0.0.0.0', server_port = 9000)
