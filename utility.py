import torch
import timm
from torchinfo import summary

def model_information(model_name , num_classes, img_size,train_batch_size):

    print('='*50)
    print('Model information')
    print('='*50)

    model_model= timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    # Print a summary using torchinfo (uncomment for actual output)
    summary(model=model_model,
            input_size=(train_batch_size, 3, img_size, img_size), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    print('='*50)
    print('Strat training')
    print('='*50)

    # Clear GPU memory by deallocating tensors
    torch.cuda.empty_cache()

    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


