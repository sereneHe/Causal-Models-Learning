# read observations
# the stock-market.txt is generated by
"""
# Load stock-market data
load_path = 'setting6.mat'
load_data = sio.loadmat(load_path)
seq=flatten(load_data['seq_d0'].tolist())
"""
ts = readdlm( "stock-market.txt" );