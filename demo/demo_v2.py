import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM
from datasets import load_dataset
import torch

def initialize_models():
    # unlearned model init
    model_name = "cais/Zephyr_RMU"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer1 = AutoTokenizer.from_pretrained(model_name)
    tokenizer1.padding_side = "left"
    tokenizer1.pad_token_id = tokenizer1.eos_token_id

    # dpo model init
    checkpoint_path = "neeljaycs/zephyr-7b-beta-refusal-dpo"
    model2 = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="cuda:0",
        torch_dtype=torch.float16
    )
    tokenizer2 = AutoTokenizer.from_pretrained(model_name)
    tokenizer2.padding_side = "left"
    tokenizer2.pad_token_id = tokenizer2.eos_token_id

    return model1, tokenizer1, model2, tokenizer2

def generate_text(prompt, model, tokenizer, max_length=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda:0")
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model1, tokenizer1, model2, tokenizer2 = initialize_models()

def compare_models(prompt, temperature=0.7, max_length=512):
    response1 = generate_text(prompt, model1, tokenizer1, max_length, temperature)
    response2 = generate_text(prompt, model2, tokenizer2, max_length, temperature)
    return response1, response2

with gr.Blocks() as demo:
    gr.Markdown("# Model Comparison")
    gr.Markdown("Try to jailbreak the models!")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Enter your prompt", lines=3)
            temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
            max_length = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Max Length")
            submit_btn = gr.Button("Generate Responses")
    
    with gr.Row():
        with gr.Column():
            output1 = gr.Textbox(label="Zephyr-RMU Response", lines=10)
        with gr.Column():
            output2 = gr.Textbox(label="Zephyr-7b-beta-refusal-dpo Response", lines=10)
    
    submit_btn.click(
        fn=compare_models,
        inputs=[input_text, temperature, max_length],
        outputs=[output1, output2]
    )

if __name__ == "__main__":
    demo.launch(share=True)
