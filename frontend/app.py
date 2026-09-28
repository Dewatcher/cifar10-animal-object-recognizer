import gradio as gr
import random

def detect_hazards(image):
    fake_results = [
        "🚗 Car detected (92% confidence)",
        "🚶 Pedestrian detected (87% confidence)",
        "🐕 Animal detected (78% confidence)"
    ]
    num_results = random.randint(1, 3)
    return "\n".join(random.sample(fake_results, num_results))

demo = gr.Interface(
    fn=detect_hazards,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Road Hazard Detector (Prototype)",
    description="Upload a road photo. Currently using fake results for testing."
)

demo.launch()
