from gcerlib.configs.config import get_config
# from gcerlib.data.dataset import get_dataset
# from gcerlib.model import get_model
# from gcerlib.training import get_trainer
# from gcerlib.evaluation import evalutate

def main(config):
    
    print(config)
    
    dataset = get_dataset(config) # np array prob? depends on config
    model = get_model(config) # returns torch.nn.Module
    trainer = get_trainer(dataset, model, config) # fully supervised only supported initially
    
    trainer.train(model, dataset, config)

    
    


if __name__ == '__main__':
    config = get_config()
    main(config)