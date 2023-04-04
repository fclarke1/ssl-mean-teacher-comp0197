from mean_teacher import mean_teacher_train
from supervised import supervised_train
from evaluate import evaluate_all_models
import sys


evaluate = False  # set default value for evaluate

if len(sys.argv) > 1:
    if sys.argv[1] == 'evaluate':
        evaluate = True
    else:
        supervised_float = float(sys.argv[1])
else:
    print("\n NOTE: Write the % of labelled data of the Mean Teacher model you want to train!! Example: !python main_train.py 0.05 or !python main_train.py evaluate")

if evaluate:
    print("Evaluation mode enabled")
    evaluate_all_models()
else:
    supervised_pt = "{:.0f}".format(supervised_float * 100)
    unsupervised_pt = "{:.0f}".format((1 - supervised_float) * 100)

    print("\nWe will train 3 models")
    print("1: Supervised only with {}% of the dataset".format(supervised_pt))
    print("2: Semi-supervised with Mean Teacher and a split of {}% labelled / {}% unlabelled data".format(supervised_pt, unsupervised_pt))
    print("3: Supervised only with 100% of the dataset")

    # Train the base line model, the SSL mode, and the upper bound 100% labelled model
    mean_teacher_train(supervised_float)
    supervised_train(supervised_float)
    supervised_train(1.0)

