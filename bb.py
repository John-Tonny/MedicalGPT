import gradio as gr
 
def square_number(input_number):
    return input_number ** 2
 
custom_slider = gr.inputs.Slider(minimum=0, maximum=10, step=0.1, default=5, label="Select a number:")
iface = gr.Interface(fn=square_number, inputs=custom_slider, outputs="text", description="Enter a number and get its square.")
iface.launch(server_name='0.0.0.0', server_port=9000)
