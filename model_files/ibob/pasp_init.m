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
defaults = {'numcomputers', 16, 'numchannels', 16, 'samplesperpacket', 64};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'pasp');
munge_block(blk, varargin{:});

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
numchannels = get_var('numchannels', 'defaults', defaults, varargin{:});
samplesperpacket = get_var('samplesperpacket', 'defaults', defaults, varargin{:});

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

% if the number of channels is the same as the number of computers
% need to initialize a reorder block and square transposer
if numcomputers==numchannels,
    numsamples=numchannels;
 
    %calculate the reorder function so each data chunk has 1 channel
    map=[];
    for i=0:numcomputers/2-1,
        map=[map,i,i+numcomputers/2];
    end
    
    %check if we have a reorder block
    if(~isempty(find_system(gcb,'LookUnderMasks','all','SearchDepth',1,'Name','reorder'))),
        
        %if there is a reorder block just reset the map
        set_param([blk,'/reorder'],'map',mat2str(map));
          
    else,
        %if there is no reorder block add it
        add_block('casper_library/Reorder/reorder',[blk,'/reorder'],'Map', mat2str(map), 'n_inputs', '2', 'Position', [2860 974 2955 1046]);
        add_block('casper_library/Reorder/square_transposer',[blk,'/square_transposer'],'n_inputs', '1', 'Position', [3030 942 3110 1068]);    
        
        % rewire the sync line
        delete_line(blk,'bs_delay/1','dist_gbe/1');
        add_line(blk,'bs_delay/1','reorder/1');
        add_line(blk,'reorder/1','square_transposer/1');
        add_line(blk,'square_transposer/1','dist_gbe/1');
        
        % add the constant
        add_block('xbsIndex_r3/Constant',[blk,'/cns_reorder'],'const','1','arith_type','Boolean', 'explicit_period', 'on', 'Position', [2735 984 2805 1016]);
        add_line(blk,'cns_reorder/1','reorder/2');
        
        % rewire the data lines
        delete_line(blk,'cram0/1','Concat1/1');
        delete_line(blk,'cram1/1','Concat1/2');
        add_line(blk,'cram0/1','reorder/3');
        add_line(blk,'cram1/1','reorder/4');
        add_line(blk,'reorder/3','square_transposer/2');
        add_line(blk,'reorder/4','square_transposer/3');
        add_line(blk,'square_transposer/2','Concat1/1');
        add_line(blk,'square_transposer/3','Concat1/2');
    end
    
else,
    numsamples=numchannels/2;
    %check if we have a reorder block (if there is remove it)
    if(~isempty(find_system(gcb,'LookUnderMasks','all','SearchDepth',1,'Name','reorder'))),
        % rewire the sync line
        delete_line(blk,'bs_delay/1','reorder/1');
        delete_line(blk,'reorder/1','square_transposer/1');
        delete_line(blk,'square_transposer/1','dist_gbe/1');
        add_line(blk,'bs_delay/1','dist_gbe/1');
        
        % remove the constant
        delete_line(blk,'cns_reorder/1','reorder/2');
        
        % rewire the data lines
        delete_line(blk,'cram0/1','reorder/3');
        delete_line(blk,'cram1/1','reorder/4');
        delete_line(blk,'reorder/3','square_transposer/2');
        delete_line(blk,'reorder/4','square_transposer/3');
        delete_line(blk,'square_transposer/2','Concat1/1');
        delete_line(blk,'square_transposer/3','Concat1/2');
        add_line(blk,'cram0/1','Concat1/1');
        add_line(blk,'cram1/1','Concat1/2');
        
        % remove the constant, reorder and square transposer
        delete_block([blk,'/cns_reorder']);
        delete_block([blk,'/reorder']);
        delete_block([blk,'/square_transposer']);
    end
        
        
        
end
    
    

set_param([blk,'/dist_gbe'],'numcomputers',num2str(numcomputers),'numsamples',num2str(numsamples),'samplesperpacket',num2str(samplesperpacket));

clean_blocks(blk);

fmtstr = sprintf('IPs=%d Channels=%d Smpl/pkt=%d', numcomputers, numchannels, samplesperpacket);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
