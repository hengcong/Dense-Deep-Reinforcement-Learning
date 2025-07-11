import torch
import os
import pickle
from collections import defaultdict
from mtlsp.pedestrian.SLSTM.SocialLSTM import SocialModel
from mtlsp.pedestrian.SLSTM.grid import getSequenceGridMask
from mtlsp.pedestrian.SLSTM.helper import vectorize_seq, revert_seq, sample_gaussian_2d, getCoef


def predict_trajectory(past_data):
    # Load args (parameters)
    cwd = os.getcwd()
    subdir = 'mtlsp/pedestrian'
    model_dir = os.path.join(cwd, subdir, 'SLSTM')
    args_path = os.path.join(model_dir, 'config.pkl')
    with open(args_path, 'rb') as arg_file:
        args = pickle.load(arg_file)

    # Load checkpoints
    checkpoint_path = os.path.join(model_dir, 'SOCIALLSTM_lstm_model_6.tar')
    checkpoint = torch.load(checkpoint_path)

    # Load model
    model = SocialModel(args=args)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Process input data
    data_dict = defaultdict(list)
    for frame_id, ped_id, x, y in past_data:
        data_dict[ped_id].append((frame_id, x, y))
    seq_len = args.seq_length
    input_seq, ped_ids = [], []
    for ped_id, traj in data_dict.items():
        traj.sort()
        coords = [(x, y) for _, x, y in traj][-seq_len:]
        if len(coords) == seq_len:
            input_seq.append(coords)
            ped_ids.append(ped_id)

    num_peds = len(input_seq)
    sequence = torch.zeros(seq_len, num_peds, 2)
    for i, coords in enumerate(input_seq):
        for t in range(seq_len):
            sequence[t, i] = torch.tensor(coords[t])
    pedlist_seq = [[ped_ids[i] for i in range(num_peds)] for _ in range(seq_len)]
    dimensions = [720, 576]  # BEV window size (change if needed)
    look_up = {ped_id: i for i, ped_id in enumerate(ped_ids)}

    # Grid Mask and Vesctorization
    grids = getSequenceGridMask(sequence, dimensions, pedlist_seq,
                            args.neighborhood_size, args.grid_size, args.use_cuda)
    vectorized_seq, first_vals = vectorize_seq(sequence, pedlist_seq, look_up)

    # Init states
    hidden_states = torch.zeros(num_peds, args.rnn_size)
    cell_states = torch.zeros(num_peds, args.rnn_size) if not args.gru else None
    if args.use_cuda:
        vectorized_seq = vectorized_seq.cuda()
        grids = [g.cuda() for g in grids]
        hidden_states = hidden_states.cuda()
        if cell_states is not None:
            cell_states = cell_states.cuda()
    
    # Prediction
    # pred_len = args.pred_length  --> default as 12 steps
    pred_len = 1
    output_seq = torch.zeros(pred_len, num_peds, 2)
    last_input = vectorized_seq[-1]

    with torch.no_grad():
        for t in range(pred_len):
            grid_mask = getSequenceGridMask(last_input.unsqueeze(0).cpu(), dimensions,
                                            [ped_ids], args.neighborhood_size, args.grid_size, args.use_cuda)[0]
            if args.use_cuda:
                grid_mask = grid_mask.cuda()

            out, hidden_states, cell_states = model(
                last_input.unsqueeze(0), [grid_mask], hidden_states, cell_states,
                [ped_ids], [num_peds], None, look_up
            )

            mux, muy, sx, sy, corr = getCoef(out)
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, ped_ids, look_up)
            next_frame = torch.stack((next_x, next_y), dim=1)

            if args.use_cuda:
                next_frame = next_frame.cuda()

            output_seq[0] = next_frame
            last_input = next_frame.clone()
    abs_output_seq = revert_seq(output_seq, [ped_ids]*pred_len, look_up, first_vals)

    # Get predicted trajectory
    last_frame = max(f for f, *_ in past_data)
    predicted_traj = []
    for ped_idx, ped_id in enumerate(ped_ids):
        for t in range(pred_len):
            x, y = abs_output_seq[t, ped_idx].cpu().numpy()
            frame_id = last_frame + (t + 1) * 12
            predicted_traj.append((frame_id, ped_id, x, y))
    
    return predicted_traj



# Test
if __name__ == '__main__':
    # Load sample data
    cwd = os.getcwd()
    subdir = 'mtlsp/pedestrian'
    model_dir = os.path.join(cwd, subdir, 'SLSTM')
    # sample_file = 'bookstore_0_0.txt'
    sample_file = 'test.txt'        #  5 pedestrain only
    sample_data_path = os.path.join(model_dir, sample_file)
    
    sample_data = []
    with open(sample_data_path, 'r') as f:
        for line in f:
            frame_id, ped_id, x, y = map(float, line.strip().split())
            sample_data.append((frame_id, int(ped_id), x, y))
    
    predicted_traj = predict_trajectory(sample_data)
    print(predicted_traj)