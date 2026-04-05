import argparse
import torch

parser = argparse.ArgumentParser(description='S2-Net Training Configuration')

# ================= 1. Path Arguments =================
parser.add_argument('--data_root', type=str, required=True, 
                    help='Root directory for BOLD_interpolated data')
parser.add_argument('--sc_root', type=str, required=True, 
                    help='Root directory for HCP-YA-SC data')
# [Task 1: Sequence Labeling]
parser.add_argument('--label_csv_rl', type=str, required=False, 
                    help='Path to RL_label.csv')
parser.add_argument('--label_csv_lr', type=str, required=False, 
                    help='Path to LR_label.csv')
# [Task 2: Subject Classification]
parser.add_argument('--label_file', type=str, default=None, required=False,
                    help='[SubjCLS] Path to clinical label file (csv/xlsx)')
parser.add_argument('--dataset', type=str, default='HCPYA', required=False,
                    help='[SubjCLS] Dataset name: HCPYA, HCPA, UKB, ADNI, PPMI, NIFD')
parser.add_argument('--fallback_sc', type=str, default=None, 
                    help='Path to fallback SC .mat file')
parser.add_argument('--output_dir', type=str, default='./exp', 
                    help='Directory to save logs and checkpoints')

# ================= 2. Training Setup =================
parser.add_argument('--seed', type=int, default=1111, 
                    help='Random seed')
parser.add_argument('--folds', type=int, default=10, 
                    help='Number of cross-validation folds')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use (cuda/cpu)')

# ================= 3. Existing Model & SNN Arguments =================
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--gpu', default='0', type=int, metavar='N',
                    help='cuda device id')

parser.add_argument('--ablation', type=str, default='full', 
                    choices=['full', 'no_coupling', 'no_mask', 'no_smooth', 'raw_feature'],
                    help='Ablation study mode')

# SNN / Kuramoto Params
parser.add_argument('--hidden', default=32, type=int, metavar='N', help='Latent dim (Eq.1)')
parser.add_argument('--k', default=1.0, type=float, metavar='N', help='Coupling strength K')
parser.add_argument('--dt', default=0.1, type=float, metavar='N', help='Time step')

# Neuron Params
parser.add_argument('--low_n', default=0, type=int, metavar='N', help='SNN low_n')
parser.add_argument('--high_n', default=4, type=int, metavar='N', help='SNN high_n')
parser.add_argument('--branch', default=1, type=int, metavar='N', help='SNN branch')

# Other existing params (kept for compatibility)
# parser.add_argument('--dataset', default='HCPYA', type=str, metavar='N', help='dataset name')
parser.add_argument('--algo', default='SRNN', type=str, metavar='N', help='algorithm')
parser.add_argument('--thresh', default=0.5, type=float, metavar='N', help='threshold')
parser.add_argument('--lens', default=0.5, type=float, metavar='N', help='lens')
parser.add_argument('--decay', default=0.5, type=float, metavar='N', help='decay')
parser.add_argument('--in_size', default=4, type=int, metavar='N', help='input size')
parser.add_argument('--out_size', default=9, type=int, metavar='N', help='output size')
parser.add_argument('--phase_max', nargs='*', default=[0.0, 0.0], type=float, metavar='N')
parser.add_argument('--cycle_min', nargs='*', default=[10, 10], type=int, metavar='N')
parser.add_argument('--cycle_max', nargs='*', default=[50, 50], type=int, metavar='N')
parser.add_argument('--duty_cycle_min', nargs='*', default=[0.95, 0.95], type=float, metavar='L')
parser.add_argument('--duty_cycle_max', nargs='*', default=[1.0, 1.0], type=float, metavar='L')
parser.add_argument('--fc', nargs= '+', default=[128], type=int, metavar='N')
parser.add_argument('--num_steps', default=1301, type=int, metavar='N')
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH')
parser.add_argument('--save_path', default='', type=str, metavar='PATH')

# Use parse_args() now because we have defined all necessary arguments
args = parser.parse_args()