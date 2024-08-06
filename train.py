from trl import SFTTrainer
import datasets
from   datasets     import load_dataset
from   datasets     import Dataset
import torch
from   peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model,PeftModel, PeftConfig
import os
import pandas       as     pd
import torch.nn     as     nn
import bitsandbytes as     bnb
from   transformers import AutoModelForTokenClassification,Trainer,AutoTokenizer,AutoModel, AutoConfig, pipeline,AutoModelForCausalLM,BitsAndBytesConfig,HfArgumentParser,TrainingArguments,logging
#import wandb
from   datetime import datetime
import multiprocessing
import torch
import gc
from datasets import load_dataset,features
from trl import DPOConfig, DPOTrainer
from transformers import AutoProcessor,BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer,AutoModelForVision2Seq


processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty", do_image_splitting=False)  #padding=True, truncation=True,padding_side = 'left', 

def format(example):
    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    #max_size = processor.image_processor.size["longest_edge"]
    #example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    use_4bit = False

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            use_nested_quant = False
        )


    model_name = "HuggingFaceM4/idefics2-8b-chatty"
    
    train_dataset = load_dataset("openbmb/RLAIF-V-Dataset"     , split="train[:1%]")
    validation_dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[50:51%]")
    tr_ds=train_dataset[:20]
    train_dataset=Dataset.from_dict(tr_ds)
    tr_ds=validation_dataset[:20]
    validation_dataset=Dataset.from_dict(tr_ds)    
    train_dataset = train_dataset.map(format, remove_columns=train_dataset.column_names)
    validation_dataset = validation_dataset.map(format, remove_columns=validation_dataset.column_names)
    print('DATA LOADED') 
    # Make sure that the images are decoded, it prevents from storing bytes.
    # More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478

    f = train_dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))  # to avoid bytes
    train_dataset = train_dataset.cast(f)
    validation_dataset = validation_dataset.cast(f)
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    # LoRA attention dimension
    lora_r = 8

    # Alpha parameter for LoRA scaling
    lora_alpha = 8

    # Dropout probability for LoRA layers
    lora_dropout = 0.1


    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
                    #'.*(vision_model).*(out_proj|k_proj|q_proj|v_proj).*$'],
        bias="none",
        use_dora=False,  # True without QLORA
        init_lora_weights="gaussian",
        #task_type="CAUSAL_LM",
        )


    training_arguments = DPOConfig(
        output_dir='/content/gdrive/MyDrive/gdrive/idefics2-dpo/model_output',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        #optim="paged_adamw_32bit",
        eval_strategy="steps",
        save_steps=0,
        save_strategy="epoch",
        logging_steps=400,
        learning_rate=1e-4,
        weight_decay=1e-4,
        fp16=True,
        #fp16=False,
        #bf16=True,
        max_grad_norm=0.3,
        warmup_ratio = 0.03,
        warmup_steps=100,
        #gradient_checkpointing=True,
        #hub_model_id='',
        #push_to_hub=True,
        #deepspeed="ds_zero3.json",
        #group_by_length=True,
        lr_scheduler_type="cosine",#,
        report_to="wandb",
        run_name="no_vision_encoder_4",
        #gradient_checkpointing_kwargs={'use_reentrant':True}
        )  
    print('PARAMS SET')    
    model = Idefics2ForConditionalGeneration.from_pretrained(model_name,torch_dtype=torch.float16,quantization_config=bnb_config)#.to(DEVICE)  ,attn_implementation="flash_attention_2"
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print('MODEL LOADED')
    #
    #tokenizer=processor.tokenizer
    #tokenizer.padding_side = 'left'
    #truncation_side = 'left'
    # Set supervised fine-tuning parameters
    trainer = DPOTrainer(   model,
                            ref_model=None,
                            train_dataset=train_dataset, #shuffled_dataset, #dataset_1,
                            eval_dataset=validation_dataset,
                            max_length=32,
                            beta=0.07,
                            tokenizer=processor,
                            args=training_arguments,
                            #rpo_alpha
                            )

    trainer.train()

    trainer.model.save_pretrained('/content/gdrive/MyDrive/gdrive/idefics2-dpo/model_output') 
    print("FINISH TRAIN")                           
    del model
    del trainer

    torch.cuda.empty_cache()
    gc. collect()
    
    peft_config = PeftConfig.from_pretrained('/content/gdrive/MyDrive/gdrive/idefics2-dpo/model_output')
    model = Idefics2ForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            #load_in_8bit=True,
            return_dict=True,
            #device_map="auto",
            torch_dtype=torch.float16,
            #low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
            model,
            '/content/gdrive/MyDrive/gdrive/idefics2-dpo/model_output',
            torch_dtype=torch.float16,
            device_map="auto",
    )
    #model.eval()
    os.makedirs('/content/gdrive/MyDrive/gdrive/idefics2-dpo/model_trained', exist_ok=True)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained('/content/gdrive/MyDrive/gdrive/idefics2-dpo/model_trained')   
    merged_model.push_to_hub("SvPolina/MyModel_chatty")
    processor.push_to_hub("SvPolina/MyModel_chatty")     

if __name__ == "__main__":
    main()