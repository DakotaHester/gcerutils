import segmentation_models_pytorch as smp

def get_model(config):
    model_arch = config.model.model_arch.lower()
    
    unet_names = ['unet', 'u_net', 'u-net']
    if model_arch in unet_names:
        model = smp.Unet(
            encoder_name=config.model.model_encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            classes=config.model.classes,
            decoder_attention_type=config.model.decoder_attention_type
        )
        print(f'[MODEL] Loaded UNet model with hyperparameters: {config.model.__dict__}')
        return model
        
    unetpp_names = ['unet++', 'unetplusplus', 'unetpp']
    if model_arch in unetpp_names:
        model = smp.UnetPlusPlus(
            encoder_name=config.model.model_encoder,
            encoder_weights=config.model.encoder_weights,
            in_channels=config.model.in_channels,
            classes=config.model.classes,
            decoder_attention_type=config.model.decoder_attention_type
        )
        print(f'[MODEL] Loaded UNet++ model with hyperparameters: {config.model.__dict__}')
        return model
    
    manet_names = ['manet', 'ma-net++']
    if model_arch in manet_names:
        pass
    # so on and so forth. MUST BE TORCH.NN.MODULE