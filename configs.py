import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--training_epoch', type=int, default=1000,
                    help='Number of epochs to train')
parser.add_argument('--gru_units', type=int, default=64,
                    help='hidden units of gru')
parser.add_argument('--seq_len', type=int, default=4,
                    help='time length of inputs.')
parser.add_argument('--pre_len', type=int, default=3,
                    help='time length of prediction')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--train_rate', type=float, default=0.8,
                    help='rate of training set')
parser.add_argument('--city', type=str, default='LA',
                    help='SZ or LA')
parser.add_argument('--model_name', type=str, default='T-GCN',
                    help='model selection')
args = parser.parse_args()

#flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_integer('training_epoch', 1000, 'Number of epochs to train.')
# flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
#flags.DEFINE_integer('seq_len', 12, '  time length of inputs.')
# flags.DEFINE_integer('pre_len', 3, 'time length of prediction.')
# flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
# flags.DEFINE_integer('batch_size', 64, 'batch size.')
# flags.DEFINE_string('city', 'LA', 'SZ or LA.')
# flags.DEFINE_string('model_name', 'tgcn', 'tgcn')