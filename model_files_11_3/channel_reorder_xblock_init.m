function channel_reorder_xblock_init(numcomputers, numchannels, samplesperpacket, qdr_reorder)
% To make the xBlock smart so it won''t re-generate the
% subsystem if neither the arguments nor the scripts are
% changes, use as the following:
% cfg.source = str2func('channel_reorder_xblock_init');
% cfg.toplevel = gcb;
% cfg.depend = {'channel_reorder_xblock_init'};
% args = {};
% xBlock(cfg, args);

sync_in = xInport('sync_in');
pol0in0 = xInport('pol0in0');
pol0in1 = xInport('pol0in1');
pol1in0 = xInport('pol1in0');
pol1in1 = xInport('pol1in1');

sync_out = xOutport('sync_out');
data_out = xOutport('data_out');


%concatenate the in0's and in1's together
ch0concat = xSignal;
ch1concat = xSignal;
xBlock('Concat',[],{pol0in0,pol1in0},{ch0concat});
xBlock('Concat',[],{pol0in1,pol1in1},{ch1concat});


%calculate the reorder function so each data chunk has 1 channel (demux channels)
map=[];
for i=0:numchannels/2-1,
    map=[map,i,i+numchannels/2];
end

% reorder so each 64 bit chunk all comes from the same channel

% add the constant
cns_reorder=xSignal;
xBlock('Constant',struct('const',1,'arith_type','Boolean'),{},{cns_reorder});
reorder_sync=xSignal;
reordered_channels0=xSignal;
reordered_channels1=xSignal;
xBlock(struct('source','casper_library_reorder/reorder'),struct('Map',map,'n_inputs',2),{sync_in,cns_reorder,ch0concat,ch1concat},{reorder_sync,reordered_channels0,reordered_channels1});
alignedchannel_sync = xSignal;
alignedchannels0 = xSignal;
alignedchannels1 = xSignal;
xBlock(struct('source','casper_library_reorder/square_transposer'),struct('n_inputs',1),{reorder_sync,reordered_channels0,reordered_channels1},{alignedchannel_sync,alignedchannels0,alignedchannels1});    
alignedchannels = xSignal;
xBlock('Concat',[],{alignedchannels0,alignedchannels1},{alignedchannels});


channelsperpacket = numchannels/numcomputers;
samplesperchannel = samplesperpacket/channelsperpacket;



reorder_sync_out=xSignal;
xBlock('Delay',struct('reg_retiming','on','latency',2,'ShowName','off'),{reorder_sync_out},{sync_out});
reorder_data_out=xSignal;
xBlock('Delay',struct('reg_retiming','on','latency',2,'ShowName','off'),{reorder_data_out},{data_out});

if strcmp(qdr_reorder,'on'),
    xBlock(struct('source','casper_library_reorder/qdr_transpose'),...
           struct('idepth',log2(numchannels),'odepth',log2(samplesperchannel),'which_qdr','qdr0'),...
           {alignedchannel_sync,alignedchannels},{reorder_sync_out,reorder_data_out});
else
    reorder_en = xSignal;
    xBlock('Constant',struct('const',1,'arith_type','Boolean','ShowName','off'),{},{reorder_en});
    xBlock(struct('source','casper_library_reorder/reorder'),...
           struct('n_inputs',1,'map',makereorderarray(numcomputers,numchannels,samplesperchannel)),...
           {alignedchannel_sync, reorder_en, alignedchannels},{reorder_sync_out,[],reorder_data_out});
end





end
%if strcmp(qdr_reorder,'on'),
%fmtstr = sprintf('IPs=%d\n%s', numcomputers);
%else
%fmtstr = sprintf('IPs=%d\n%s', numcomputers, get_param([blk,'/reorder'], 'AttributesFormatString'));
%end

%set_param(blk, 'AttributesFormatString', fmtstr);
%save_state(blk, 'defaults', defaults, varargin{:});
