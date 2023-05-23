from gcerlib.configs.config import get_config
from gcerlib.data.load_data import get_dataset
# from gcerlib.data.dataset import get_dataset
from gcerlib.models import get_model
from gcerlib.training import get_trainer
# from gcerlib.evaluation import evalutate

def main(config):
        
    dataset = get_dataset(config) # np array prob? depends on config
    model = get_model(config) # returns torch.nn.Module
    trainer = get_trainer(config, dataset, model) # fully supervised only supported initially
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    config = get_config()
    main(config)