import yaml
import argparse

from .model import make_model_and_optimizer


def main(conf):
    # Load the model from the local definition
    model, _ = make_model_and_optimizer(conf['train_conf'])
    model.eval()
    # Do all the rest ..


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file')
    # ... Other eval specific args
    args = parser.parse_args()
    # Load training config
    with open(args.conf_file) as f:
        train_conf = yaml.safe_load(f)
    arg_dic = dict(vars(args))
    arg_dic['train_conf'] = train_conf
    main(arg_dic)
