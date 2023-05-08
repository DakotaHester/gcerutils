from gcerutils.config import get_config
from gcerutils.data.dataset import get_dataset
from gcerutils.model import get_model
from gcerutils.training import get_trainer
from gcerutils.evaluation import evalutate

def main(config):
    
    dataset = get_dataset(config) # np array prob? depends on config
    model = get_model(config) # returns torch.nn.Module
    trainer = get_trainer(dataset, model, config) # fully supervised only supported initially
    
    trainer.train(model, dataset, config)

    
    


if __name__ == '__main__':
    config = get_config()
    main(config)