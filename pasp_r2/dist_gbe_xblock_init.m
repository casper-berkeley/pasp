function dist_gbe_xblock_init(numcomputers, samplesperpacket,numtengbe)
% To test it, run the following commands from MATLAB console:
% cfg.source = str2func('dist_gbe_xblock_init');
% cfg.toplevel = 'simpletest/dist_gbe_xblock_init';
% args = {};
% xBlock(cfg, args);
% 
% You can edit ip_ctr_xblock_init.m to debug your script.
% 
% You can also replace the MaskInitialization code with the 
% following commands so the subsystem will be generated 
% according to the values of mask parameters.
% cfg.source = str2func('dist_gbe_xblock_init');
% cfg.toplevel = gcb;
% args = {};
% xBlock(cfg, args);
% 
% To configure the xBlock call in debug mode, in which mode,
% autolayout will be performed every time a block is added,
% run the following commands:
% cfg.source = str2func('dist_gbe_xblock_init');
% cfg.toplevel = gcb;
% cfg.debug = 1;
% args = {};
% xBlock(cfg, args);
% 
% To make the xBlock smart so it won''t re-generate the
% subsystem if neither the arguments nor the scripts are
% changes, use as the following:
% cfg.source = str2func('dist_gbe_xblock_init');
% cfg.toplevel = gcb;
% cfg.depend = {'dist_gbe_xblock_init'};
% args = {};
% xBlock(cfg, args);
% 
% See also xBlock, xInport, xOutport, xSignal, xlsub2script.

% Declare any default values for arguments you might like.
%defaults = {'numcomputers', 16, 'samplesperpacket', 64, 'numtengbe',2};
%if same_state(blk, 'defaults', defaults, varargin{:}), return, end
%check_mask_type(blk, 'dist_gbe');
%munge_block(blk, varargin{:});
%delete_lines(blk);

%numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
%samplesperpacket = get_var('samplesperpacket', 'defaults', defaults, varargin{:});
%numtengbe = get_var('numtengbe', 'defaults', defaults, varargin{:});

%load_system('pasp_lib.mdl');
%load_system('casper_library.mdl');

%% inports
sync_in = xInport('sync');
data_in = xInport('data_in');
sys_ctr = xInport('sys_ctr');

%% outports
% none
sync_delay=xSignal;
data_delay=xSignal;
xBlock('Delay',struct('reg_retiming','on','latency',3),{sync_in},{sync_delay});
xBlock('Delay',struct('reg_retiming','on','latency',3),{data_in},{data_delay});

next_packet=xSignal;
dest_ip=xSignal;
dest_port=xSignal;
discard_pkt=xSignal;

% config for ip_ctr
ip_ctr_config.source = str2func('ip_ctr_xblock_init');
ip_ctr_config.depend = {'ip_ctr_xblock_init'};
%ip_ctr_config.name = 'pie';
xBlock(ip_ctr_config,{numcomputers,'off'},{next_packet,sync_delay},{dest_ip,dest_port,discard_pkt});


const_0=xSignal;
xBlock('Constant',struct('const',0,'ShowName','off','explicit_period','on'),{},{const_0});
rst_packet_count=xSignal;
xBlock(struct('source','xps_library/software register','name','rst_pkt_count'),struct('io_dir','From Processor'),{const_0},{rst_packet_count});
rst_slice=xSignal;
xBlock('Slice',struct('nbits',1,'boolean_output','On','ShowName','off'),{rst_packet_count},{rst_slice});

numcomputers_const=xSignal;
samplesperpacket_const=xSignal;
numtengbe_const=xSignal;
xBlock('Constant',struct('const',numcomputers,'n_bits',log2(numcomputers)+1,'bin_pt',0,'ShowName','off','explicit_period','on'),{},{numcomputers_const});
xBlock('Constant',struct('const',samplesperpacket,'n_bits',log2(samplesperpacket)+1,'bin_pt',0,'ShowName','off','explicit_period','on'),{},{samplesperpacket_const});
xBlock('Constant',struct('const',numtengbe,'n_bits',log2(numtengbe)+1,'bin_pt',0,'ShowName','off','explicit_period','on'),{},{numtengbe_const});

next_packet_inputs=cell(1,numtengbe);

for i=0:numtengbe-1,

    packetizer_dout=xSignal;
    packetizer_valid=xSignal;
    packetizer_eof=xSignal;
    tengbe_id=xSignal;
    xBlock('Constant',struct('const',i,'n_bits',log2(numtengbe)+1,'bin_pt',0,'ShowName','off','explicit_period','on'),{},{tengbe_id});
    xBlock('MCode',struct('mfname','packetizer.m'),...
           {sync_delay,data_delay,sys_ctr,samplesperpacket_const,numcomputers_const,numtengbe_const,tengbe_id},...
           {packetizer_dout,packetizer_valid,packetizer_eof});
    

    delay_dout=xSignal;
    delay_valid=xSignal;
    delay_eof=xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',2),{packetizer_dout},{delay_dout});
    xBlock('Delay',struct('reg_retiming','on','latency',2),{packetizer_valid},{delay_valid});
    xBlock('Delay',struct('reg_retiming','on','latency',2),{packetizer_eof},{delay_eof});

    ten_GbE_rst=xSignal;
    xBlock('Constant',struct('const',0,'arith_type','Boolean','ShowName','off','explicit_period','on'),{},{ten_GbE_rst});
    eof_andnot=xSignal;
    xBlock('Expression',struct('expression','a&~b'),{delay_eof,discard_pkt},{eof_andnot});
    discard_and=xSignal;
    xBlock('Logical',struct('logical_function','AND'),{discard_pkt,delay_eof},{discard_and});

    ten_GbE_ack=xSignal;
    xBlock('Constant',struct('const',0,'arith_type','Boolean','ShowName','off','explicit_period','on'),{},{ten_GbE_ack});

    % add delays for all inputs to the 10gbe block
    tengbelatency = 2;
    ten_GbE_rst_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{ten_GbE_rst},{ten_GbE_rst_delay});
    delay_dout_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{delay_dout},{delay_dout_delay});
    delay_valid_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{delay_valid},{delay_valid_delay});
    dest_ip_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{dest_ip},{dest_ip_delay});
    dest_port_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{dest_port},{dest_port_delay});
    eof_andnot_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{eof_andnot},{eof_andnot_delay});
    discard_and_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{discard_and},{discard_and_delay});
    ten_GbE_ack_delay = xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',tengbelatency,'ShowName','off'),{ten_GbE_ack},{ten_GbE_ack_delay});

    %xBlock(struct('source','xps_library/ten_GbE','name',['ten_GbE',num2str(i)]),...
    %       struct('port',['ROACH:',num2str(i)]),{ten_GbE_rst_delay,delay_dout_delay,delay_valid_delay,dest_ip_delay,dest_port_delay,eof_andnot_delay,...
    %           discard_and_delay,ten_GbE_ack_delay},{});
    xBlock(struct('source','xps_library/ten_Gbe_v2','name',['ten_Gbe_v2',num2str(i)]),...
           struct('flavour', 'sfp+', 'slot', '0',...
                  'port_r2_sfpp', num2str(i), 'large_frames', 'on'),...
           {ten_GbE_rst_delay,delay_dout_delay,delay_valid_delay,dest_ip_delay,dest_port_delay,eof_andnot_delay,...
               discard_and_delay,ten_GbE_ack_delay},{});
    
    currentcount=xSignal;
    xBlock('Counter',struct('n_bits',32,'arith_type','Unsigned','cnt_type','Free Running','rst','On','en','On'),...
           {rst_slice,eof_andnot_delay},{currentcount});
    currentcount_delay=xSignal;
    xBlock('Delay',struct('reg_retiming','on','latency',2,'ShowName','off'),{currentcount},{currentcount_delay});
    xBlock(struct('source','xps_library/software register','name',['reg_packet_count',num2str(i)]),struct('io_dir','To Processor'),...
           {currentcount_delay},{});

    next_packet_inputs{i+1}=delay_eof;


end

xBlock('Logical',struct('logical_function','OR','inputs',numtengbe),next_packet_inputs,{next_packet});

%xlAddTerms(gcs);
