
import argparse
import os
import keras.backend as K

from copy import copy
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam

from utils.build_model import build_model, NET_CONFIGS
from utils.data_reading import flip_actions
from utils.data_reading import load_runs, batch_generator

# TODO: if model is provided get frame size from it
# TODO: add callbacks to model.fit: 2) testing environment ???
# TODO: dirs with data to command line arguments??
# TODO: ranking loss


parser = argparse.ArgumentParser(description="Run commands",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_model', type=str, default=None,
                    help="Saved model filename. If not provided  new model is created")
parser.add_argument('--out_model', type=str,
                    default='models/model_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
                    help="Filename for trained model")
parser.add_argument('--loss', type=str, default='cross_entropy',
                    help="Type of loss function: policy_loss, value_softmax or cross_entropy.")
parser.add_argument('--save_period', type=int, default=2, help="Interval (number of epochs) between checkpoints.")
parser.add_argument('--network', type=str, default='small_cnn', help="Network architecture as defined in net_config.py.")
parser.add_argument('--n_frames', type=int, default=4, help="Number of frames to stack to form stats.")
parser.add_argument('--width', type=int, default=84,
                    help="Width of frame for network (has no effect if downsample was provided).")
parser.add_argument('--height', type=int, default=84,
                    help="Height of frame network (has no effect if downsample was provided).")
parser.add_argument('-d', '--downsample', type=float, default=2.,
                    help="Factor of downsampling image "
                         "(for example 2 means make image 2 times smaller, saving aspect ratio).")
parser.add_argument('--batch', type=int, default=256, help="Number of samples per batch.")
parser.add_argument('--samples_per_epoch', type=int, default=100000, help="Number of samples per epoch.")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs.")
parser.add_argument('--augment', action='store_true', help="Augment images.")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
parser.add_argument('--weight_runs', action='store_true',
                    help="Weight runs according to reward obtained in run.")
parser.add_argument('--norm', action='store_true',
                    help="Do value normalization per state or not. See file loss.py for details of implementation")
parser.add_argument('--norm_coeff', type=float, default=1,
                    help="Normalization coefficient. See file loss.py for details of implementation")
parser.add_argument('--entropy', type=float, default=0.001, help="Entropy coefficient for policy loss.")
parser.add_argument('--flip', action='store_true', help="Flip image and action vertically.")
parser.add_argument('--color', action='store_true', help="Process color images instead of grayscale.")
parser.add_argument('--min_run_score', type=float, default=None, help="Minimum score in run to process run.")
parser.add_argument('--generator_workers', type=int, default=1, help="Number of workers to generate data.")
args = parser.parse_args()


def run_training():
    dirs = ['/home/dd210/Desktop/pacman/pacman_sasha_temp.tar.gz']

    # create save dir, if already exist raise error
    if not os.path.exists(args.out_model):
        os.makedirs(args.out_model)
    else:
        raise ValueError('path {} already exist, use other out_model path'.format(args.out_model))

    # save experiment info
    info_fname = os.path.join(args.out_model, 'experiment_info.txt')
    with open(info_fname, 'wb') as f:
        for k, v in args.__dict__.iteritems():
            f.write('{}: {}\n'.format(k, v))

    save_path = os.path.join(args.out_model, 'model_epoch{epoch}.h5')
    save_callback = ModelCheckpoint(save_path, period=args.save_period)

    load_runs_fn = lambda x: load_runs(x, args.height, args.width, args.downsample, args.min_run_score)
    runs = load_runs_fn([dirs[0]])

    # get info from runs
    num_actions = len(runs[0]['action_visits'][0])
    frame_dims = runs[0]['frames'][0].shape[:2]
    if args.flip:
        flip_map = flip_actions(runs[0]['action_meanings'])
    else:
        flip_map = None

    # loss function
    if args.loss == 'cross_entropy':
        loss_fn = categorical_crossentropy
        metrics = ['accuracy']
    else:
        raise ValueError('Unknown loss'.format(args.loss))

    if args.in_model:
        print 'loading model from file {}'.format(args.in_model)
        model = load_model(args.in_model, custom_objects={'loss_fn': loss_fn})
        # set lr for loaded model
        K.set_value(model.optimizer.lr, args.lr)
        # set loss function
        model.loss_functions = [loss_fn]
        model.metrics = metrics
    else:
        net_params = copy(NET_CONFIGS[args.network])
        net_params['n_actions'] = num_actions
        net_params['state_size'] = [args.n_frames*(3**args.color)] + list(frame_dims)
        print 'state size: {}'.format(net_params['state_size'])
        print 'n_actions: {}'.format(net_params['n_actions'])
        model = build_model(**net_params)
        #model = build_model(state_size)
        opt = Adam(args.lr)
        model.compile(optimizer=opt,
                      loss=loss_fn,
                      metrics=metrics)

    model.summary()

    try:
        model.fit_generator(
            generator=batch_generator(
                (load_runs_fn, dirs), num_actions, args.loss, args.augment, args.n_frames,
                flip_map, not args.color, args.weight_runs, args.batch
            ),
            samples_per_epoch=(args.samples_per_epoch/args.batch)*args.batch,
            nb_epoch=args.epochs,
            callbacks=[save_callback],
            nb_worker=args.generator_workers,
        )
    finally:
        model.save(os.path.join(args.out_model, 'model_final.h5'))


if __name__ == '__main__':
    run_training()