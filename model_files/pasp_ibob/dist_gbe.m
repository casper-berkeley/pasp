function dist_gbe(blk, varargin)
% Initialize and configure the reorder block.
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
check_mask_type(blk, 'reorder');
munge_block(blk, varargin{:});

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
numsamples = get_var('numsamples', 'defaults', defaults, varargin{:});
samplesperpacket = get_var('samplesperpacket', 'defaults', defaults, varargin{:});

set_param([gcb,'/reorder'],map,makereorderarray(numcomputers, numsamples, samplesperpacket));

clean_blocks(blk);

fmtstr = sprintf('IPs=%d', numcomputers);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
