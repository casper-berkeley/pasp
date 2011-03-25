function pasp_init(blk, varargin)
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
defaults = {'numcomputers', 16, 'numchannels', 16, 'numtengbe', 4, 'samplesperpacket', 64, 'qdr_reorder', 'off'};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'pasp');
munge_block(blk, varargin{:});

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
numchannels = get_var('numchannels', 'defaults', defaults, varargin{:});
numtengbe = get_var('numtengbe', 'defaults', defaults, varargin{:});
samplesperpacket = get_var('samplesperpacket', 'defaults', defaults, varargin{:});
qdr_reorder = get_var('qdr_reorder', 'defaults', defaults, varargin{:});

if numcomputers>numchannels,
    error('The number of IPs cannot be greater than the number of real channels');
end

% number of complex channels
logcomchannels = log2(2*numchannels);

% set up pfbs and ffts with the appropriate number of channels
set_param([blk,'/pfb_fir_real'],'PFBSize',num2str(logcomchannels));
set_param([blk,'/pfb_fir_real1'],'PFBSize',num2str(logcomchannels));
set_param([blk,'/fft_wideband_real'],'FFTSize',num2str(logcomchannels));
set_param([blk,'/fft_wideband_real1'],'FFTSize',num2str(logcomchannels));

set_param([blk,'/scale_ctr'],'numchannels',num2str(numchannels));
    
    
set_param([blk,'/channel_reorder'],'numcomputers',num2str(numcomputers),'numchannels',num2str(numchannels),'samplesperpacket',num2str(samplesperpacket),'qdr_reorder',qdr_reorder);
set_param([blk,'/dist_gbe'],'numcomputers',num2str(numcomputers),'samplesperpacket',num2str(samplesperpacket),'numtengbe',num2str(numtengbe));

clean_blocks(blk);

fmtstr = sprintf('IPs=%d Channels=%d Smpl/pkt=%d', numcomputers, numchannels, samplesperpacket);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
