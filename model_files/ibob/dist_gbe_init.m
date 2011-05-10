function dist_gbe_init(blk, varargin)
% Initialize and configure the packet distributer block.
%
% reorder_init(blk, varargin)
%
% blk = The block to be initialize.
% varargin = {'varname', 'value', ...} pairs
% 
% Valid varnames for this block are:
% map = The desired output order.
% map_latency = The latency of a map block.
% bram_latency = The latency of a BRAM block.
% n_inputs = The number of parallel inputs to be reordered.
% double_buffer = Whether to use two buffers to reorder data (instead of
%                 doing it in-place).

% Declare any default values for arguments you might like.
defaults = {'numcomputers', 16, 'numsamples', 16, 'samplesperpacket', 64};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'dist_gbe');
munge_block(blk, varargin{:});

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
numsamples = get_var('numsamples', 'defaults', defaults, varargin{:});
samplesperpacket = get_var('samplesperpacket', 'defaults', defaults, varargin{:});

%update the reorder ordering
set_param([blk,'/reorder'],'map',mat2str(makereorderarray(numcomputers, numsamples, samplesperpacket)));

%update the counters when samplesperpacket changes
set_param([blk,'/cnt_mux0'], 'n_bits', num2str(log2(samplesperpacket)+1));
set_param([blk,'/cns_mux_rel0'], 'const', num2str(2*samplesperpacket-1-1), 'n_bits', num2str(log2(samplesperpacket)+1));
set_param([blk,'/bs_delay7'], 'latency', num2str(samplesperpacket));
% add an extra sample in the pulse extenders for the system counter value
% and packet number
set_param([blk,'/pulse_ext_10GbE0'], 'pulse_len', num2str(samplesperpacket+1+1));
set_param([blk,'/pulse_ext_10GbE1'], 'pulse_len', num2str(samplesperpacket+1+1));

% set the id counters
set_param([blk,'/counter_id_odd'], 'cnt_to', num2str(numcomputers-1));
set_param([blk,'/counter_id_even'], 'cnt_to', num2str(numcomputers));

%update ip counter (num computers /2 for each)
set_param([blk,'/ip_ctr'],'numcomputers',num2str(numcomputers));


clean_blocks(blk);



fmtstr = sprintf('IPs=%d\n%s', numcomputers, get_param([blk,'/reorder'], 'AttributesFormatString'));
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
