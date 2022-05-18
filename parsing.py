"""
The parsing functions for the argument input.
"""
import os
import pickle
from argparse import ArgumentParser, Namespace

def gridless_parsing():

    # Model arguments

    parser = ArgumentParser(description='Gridless argument')
    parser.add_argument('--name', default = 'default', dest='name', type=str, help='Model Name')
    parser.add_argument('--d_model', default=128, type=int, help='Hidden Space dimension')
    parser.add_argument('--n_layers', default=2, type=int, help='Layer Number of MultiHeadAttention')
    parser.add_argument('--input', default='Bin', type=str, help='Data Input Type. [Scalar, Bin]')
    parser.add_argument('--use_latent_path', default=False, action='store_true',
                        help="")
    parser.add_argument('--d_type', default = 'grpe_5', type = str, help = '[str, grpe_1, grpe_5]')
    parser.add_argument('--use_anp', default=False, action='store_true', help="")

    # Data arguments
    parser.add_argument('--source', default='bindingDB', type=str, help="One of bindingDB, davis, kiba")
    parser.add_argument('--folder', default=None, type=str, help="data path")
    parser.add_argument('--n_bins', default=32, help='Number of histogram bins for binding affinity')
    parser.add_argument('--seq_len', default=512, type=int, help='Ligand Count per Target. Sequence Length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="")
    parser.add_argument('--withtrain', default='no', type=str, help='Fewshot test data are trained of not')
    parser.add_argument('--uniprot' , default=None, type=str, help='training target uniprot')

    # Training argument
    parser.add_argument('--select_by_loss', default=False, action='store_true',
                        help="")
    parser.add_argument('--metric', type=str, default='r2',
                        help="")
    parser.add_argument('--lr', type=float, default='1e-3',
                        help="")
    parser.add_argument('--n_epochs', type=int, default=2000,
                        help="")

    # Utility argument
    args = parser.parse_args()

    return args

def moltrans_parsing():

    parser = ArgumentParser(description='MolTrans argument')

    # DenseNet Model arguments
    parser.add_argument('--exemplar_len', type=int, default=32,
                        help="")
    parser.add_argument('--scale_down_ratio', type=float, default=0.25,
                        help="")
    parser.add_argument('--growth_rate', type=int, default=20,
                        help="")
    parser.add_argument('--transition_rate', type=float, default=0.5,
                        help="")
    parser.add_argument('--num_dense_blocks', type=int, default=4,
                        help="")
    parser.add_argument('--kernal_dense_size', type=int, default=3,
                        help="")

    # Encoder Model arguments
    parser.add_argument('--intermediate_size', type=int, default=1536,
                        help="")
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="")
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1,
                        help="")
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                        help="")
    parser.add_argument('--flat_dim', type=int, default=78192,
                        help="")

    # Dataset argument
    parser.add_argument('--source', type=str, help='One of bindingDB, davis, kiba')
    parser.add_argument('--folder', type=str, help='Directory for Binding DB')
    parser.add_argument('--batch_size', type=int, default=64,
                        help="")
    parser.add_argument('--input_dim_drug', type=int, default=23532,
                        help="")
    parser.add_argument('--input_dim_target', type=int, default=16693,
                        help="")
    parser.add_argument('--max_drug_seq', type=int, default=50,
                        help="")
    parser.add_argument('--max_protein_seq', type=int, default=545,
                        help="")
    parser.add_argument('--emb_size', type=int, default=384,
                        help="")
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help="")

    # Utility argument
    parser.add_argument('--select_by_loss', type=bool, default=True)
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs to train.')
    args = parser.parse_args()

    return args

def deepdta_parsing():

    parser = ArgumentParser(description='DeepDTA')
    # for model
    parser.add_argument('--seq_window_length', type=int, default=8, help='sequence filter lengths')
    parser.add_argument('--smi_window_length', type=int, default=4, help='smiles filter lengths')
    parser.add_argument('--num_window', type=int, default=32, help='number of motif filters corresponding to length list. (ex, --num_windows 100)')
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Length of input sequences.')
    parser.add_argument('--max_smi_len', type=int, default=100, help='Length of input sequences.')
    parser.add_argument('--K', type=int, default=128, help='Number of meta learning sample data')

    # for learning
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--meta_lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--inner_step', type=int, default=1, help='Initial learning rate.')

    # fot data
    parser.add_argument('--source', type=str, help='One of bindingDB, davis, kiba')
    parser.add_argument('--folder', type=str, help='Directory for Binding DB')

    args = parser.parse_args()

    return args

