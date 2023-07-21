from functions.mean_teacher import mean_teacher_train
from functions.supervised import supervised_train
from functions.evaluate import evaluate_all_models
import argparse


def main(args):
    # evaluate models
    if args.evaluate:
        print("Evaluation mode enabled")
        evaluate_all_models()
    # if evaluating then not going to train any models even if inputs are given
    else:
        # train 100% labelled supervised model M100
        if args.train_100pct_model:
            supervised_train(1.0)
        # train SSL model if given a labelled_pct
        if args.train_labelled_pct != None:
            supervised_float = args.train_labelled_pct
            assert 0 < supervised_float < 1, 'ERROR: labelled_pct needs to be in (0,1)'
            
            # train with validated inputs
            mean_teacher_train(supervised_float)
            supervised_train(supervised_float)
    print('Program Complete...')


if __name__=='__main__':
    # parse command line inputs
    parser = argparse.ArgumentParser(description="script to use for training or evaluating models")
    parser.add_argument("--train_labelled_pct", type=float, help="if training: percent of data to be labelled for model, eg. =0.25 will train model M25L and M25")
    parser.add_argument("--train_100pct_model", action="store_true", help='train 100% supervised model M100')
    parser.add_argument("--evaluate", action='store_true', help="evaluate all models", required=True)
    args = parser.parse_args()
    
    main(args)
