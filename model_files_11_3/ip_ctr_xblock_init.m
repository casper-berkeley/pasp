function ip_ctr_xblock_init(numcomputers,bram_table)
% This is a generated function based on subsystem:
%     simpletest/Subsystem
% Though there are limitations about the generated script, 
% the main purpose of this utility is to make learning
% Sysgen Script easier.
% 
% To test it, run the following commands from MATLAB console:
% cfg.source = str2func('Subsystem');
% cfg.toplevel = 'simpletest/Subsystem';
% args = {};
% xBlock(cfg, args);
% 
% You can edit Subsystem.m to debug your script.
% 
% You can also replace the MaskInitialization code with the 
% following commands so the subsystem will be generated 
% according to the values of mask parameters.
% cfg.source = str2func('Subsystem');
% cfg.toplevel = gcb;
% args = {};
% xBlock(cfg, args);
% 
% To configure the xBlock call in debug mode, in which mode,
% autolayout will be performed every time a block is added,
% run the following commands:
% cfg.source = str2func('Subsystem');
% cfg.toplevel = gcb;
% cfg.debug = 1;
% args = {};
% xBlock(cfg, args);
% 
% To make the xBlock smart so it won''t re-generate the
% subsystem if neither the arguments nor the scripts are
% changes, use as the following:
% cfg.source = str2func('Subsystem');
% cfg.toplevel = gcb;
% cfg.depend = {'Subsystem'};
% args = {};
% xBlock(cfg, args);
% 
% See also xBlock, xInport, xOutport, xSignal, xlsub2script.


%% inports
count_in = xInport('Count');
reset_in = xInport('Reset');

%% outports
ip_outputport = xOutport('IP');
ip_out=xSignal;
ip_outputport.bind(ip_out);

port_out = xOutport('Port');
port_slice = xSignal;
xBlock('Slice',struct('nbits',16,'mode','Lower Bit Location + Width'),{port_slice},{port_out});

discard_out = xOutport('Discard_pkt');

%% diagram

current_index = xSignal;

% create index counter
xBlock('Counter', ...
       struct('n_bits', ceil(log2(numcomputers)),'arith_type', 'Unsigned','cnt_type', 'Free Running', 'rst', 'On', 'en', 'On'), ...
       {reset_in,count_in}, ...
       {current_index});

const0 = xSignal;
xBlock('Constant',...
       struct('const',0,'arith_type','Unsigned','n_bits',32,'bin_pt',0,'ShowName','off'),...
       {},{const0});
                          
xBlock('Relational',...
       [],...
       {ip_out,const0},...
       {discard_out});

if numcomputers>32 | strcmp(bram_table,'on'),
    data_in = xSignal;
    xBlock('Constant', struct('n_bits',32,'arith_type','Unsigned'),{},{data_in});

    write_enable = xSignal;
    xBlock('Constant', struct('arith_type','Boolean','const',0),{},{write_enable});

    xBlock(struct('source','xps_library/Shared BRAM','name','ip_table'),...
           struct('arith_type','Unsigned','addr_width',11,'init_vals',[1:numcomputers]),...
           {current_index,data_in,write_enable},...
           {ip_out});
    xBlock(struct('source','xps_library/Shared BRAM','name','port_table'),...
           struct('arith_type','Unsigned','addr_width',11,'init_vals',[1:numcomputers]),...
           {current_index,data_in,write_enable},...
           {port_slice});
                  
else,  
    ip_mux_inputs=cell(1,numcomputers+1);
    ip_mux_inputs{1}=current_index;
    port_mux_inputs=cell(1,numcomputers+1);
    port_mux_inputs{1}=current_index;

    for i=1:numcomputers,
        ip_constant=xSignal;
        port_constant=xSignal;
    
        ip_reg=xSignal;
        ip_mux_inputs{i+1}=ip_reg;
        port_reg=xSignal;
        port_mux_inputs{i+1}=port_reg;
        

        xBlock(struct('source','xps_library/software register','name',['reg_ip',num2str(i-1)]),...
               struct('io_dir','From Processor'),...
               {ip_constant},{ip_reg});
        xBlock('Constant',struct('const',4000+i,'bin_pt',0),{},{ip_constant});

        xBlock(struct('source','xps_library/software register','name',['reg_port',num2str(i-1)]),...
               struct('io_dir','From Processor'),...
               {port_constant},{port_reg});
        xBlock('Constant',struct('const',4000+i,'bin_pt',0),{},{port_constant});

    end

    ip_mux=xBlock(struct('source','Mux','name','ip_mux'),...
                  struct('inputs',numcomputers, 'Latency', '3'),...
                  ip_mux_inputs,....
                  {ip_out});
    port_mux=xBlock(struct('source','Mux','name','port_mux'),...
                    struct('inputs',numcomputers, 'Latency', '3'),...
                    port_mux_inputs,...
                    {port_slice});


end

