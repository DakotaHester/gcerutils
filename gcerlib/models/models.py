import segmentation_models_pytorch as smp
from torchinfo import summary
import os

def get_model(config):
    model_arch = config.model.model_arch.lower()
    
    if model_arch in ['unet', 'u_net', 'u-net']:
        model = smp.Unet(
            encoder_name=config.model.model_encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            classes=config.model.classes,
            decoder_attention_type=config.model.decoder_attention_type
        )
        
        
    elif model_arch in ['unet++', 'unetplusplus', 'unetpp']:
        model = smp.UnetPlusPlus(
            encoder_name=config.model.model_encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            classes=config.model.classes,
            decoder_attention_type=config.model.decoder_attention_type
        )
    
    
    elif model_arch in ['dlv3+', 'deeplabv3+', 'deeplabv3plus', 'dlv3p']:
        model = smp.DeepLabV3Plus(
            encoder_name=config.model.model_encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            classes=config.model.classes,
            # decoder_attention_type=config.model.decoder_attention_type
        )
    
    elif model_arch in ['manet', 'ma-net++']:
        pass
    # so on and so forth. MUST BE TORCH.NN.MODULE
    
    else:
        raise NotImplementedError(f'Model architecture {config.model.model_arch} not implemented/recognized')
    
    print(f'[MODEL] Loaded {config.model.model_arch} model with hyperparameters: {config.model.__dict__}')
    
    model_summary = summary(model, input_size=(config.training.batch_size, config.model.in_channels, config.dataset.image_size, config.dataset.image_size))
    # print(model_summary)
    with open(os.path.join(config.out_path, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(str(model_summary))
    return model