function ip_ctr_init(blk, varargin)
% Initialize and configure the ip counter block.
%
% reorder_init(blk, varargin)
%
% blk = The block to be initialize.
% varargin = {'varname', 'value', ...} pairs
% 
% Valid varnames for this block are:


% Declare any default values for arguments you might like.
defaults = {'numcomputers', 8};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'ip_ctr');
munge_block(blk, varargin{:});

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
% calculate the number of odd and number of even ips
numcomputers=numcomputers/2;

even_x_offset=500;

delete_lines(blk);
reuse_block(blk, 'Reset', 'built-in/inport', 'Position', [65    25   95    40], 'Port', '3');

reuse_block(blk, 'Count_Odd', 'built-in/inport', 'Position', [75    60   105    75], 'Port', '1');
reuse_block(blk, 'Count_Even', 'built-in/inport', 'Position', [75+even_x_offset    60   105+even_x_offset    75], 'Port', '2');

reuse_block(blk, 'Counter_Odd', 'xbsIndex_r4/Counter', 'n_bits',num2str(log2(numcomputers)), 'arith_type', 'Unsigned', ...
    'cnt_type', 'Free Running', 'rst', 'On', 'en', 'On', 'Position', [125    47   170    73]);
reuse_block(blk, 'IPMux_Odd', 'xbsIndex_r4/Mux','inputs', num2str(numcomputers), 'Latency', '3', 'Position', [260    40   280   40+50*numcomputers]);
reuse_block(blk, 'PortMux_Odd', 'xbsIndex_r4/Mux','inputs', num2str(numcomputers), 'Latency', '3', 'Position', [260    40+60*numcomputers   280   40+110*numcomputers]);

reuse_block(blk, 'Delay_Even', 'xbsIndex_r4/Delay', 'latency', '4', 'Position', [410    43   435    67]);
reuse_block(blk, 'Counter_Even', 'xbsIndex_r4/Counter', 'n_bits',num2str(log2(numcomputers)), 'arith_type', 'Unsigned', ...
    'cnt_type', 'Free Running', 'rst', 'On', 'en', 'On', 'Position', [125+even_x_offset    47   170+even_x_offset    73]);
reuse_block(blk, 'IPMux_Even', 'xbsIndex_r4/Mux','inputs', num2str(numcomputers), 'Latency', '3', 'Position', [260+even_x_offset    40   280+even_x_offset   40+50*numcomputers]);
reuse_block(blk, 'PortMux_Even', 'xbsIndex_r4/Mux','inputs', num2str(numcomputers), 'Latency', '3', 'Position', [260+even_x_offset    40+60*numcomputers   280+even_x_offset   40+110*numcomputers]);

reuse_block(blk, 'IP_Odd', 'built-in/outport', 'Position', [305   233   335   247], 'Port', '1');
reuse_block(blk, 'Port_Odd', 'built-in/outport', 'Position', [305   283   335   297], 'Port', '2');
reuse_block(blk, 'IP_Even', 'built-in/outport', 'Position', [305+even_x_offset   233   335+even_x_offset   247], 'Port', '3');
reuse_block(blk, 'Port_Even', 'built-in/outport', 'Position', [305+even_x_offset   283   335+even_x_offset   297], 'Port', '4');

add_line(blk, 'Reset/1', 'Counter_Odd/1');
add_line(blk, 'Count_Odd/1', 'Counter_Odd/2');
add_line(blk, 'Counter_Odd/1', 'IPMux_Odd/1');
add_line(blk, 'IPMux_Odd/1', 'IP_Odd/1');
add_line(blk, 'PortMux_Odd/1', 'Port_Odd/1');
add_line(blk, 'Counter_Odd/1', 'PortMux_Odd/1');

add_line(blk, 'Reset/1', 'Delay_Even/1')
add_line(blk, 'Delay_Even/1', 'Counter_Even/1');
add_line(blk, 'Count_Even/1', 'Counter_Even/2');
add_line(blk, 'Counter_Even/1', 'IPMux_Even/1');
add_line(blk, 'IPMux_Even/1', 'IP_Even/1');
add_line(blk, 'PortMux_Even/1', 'Port_Even/1');
add_line(blk, 'Counter_Even/1', 'PortMux_Even/1');

start=1;
for i=start:2:2*numcomputers,
    %add ip registers
    reuse_block(blk,['reg_ip',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+ceil(i/2)*45 185 75+ceil(i/2)*45]);
    reuse_block(blk,['IP_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', num2str(4000+i),...
        'Position',[25 45+ceil(i/2)*45 55 75+ceil(i/2)*45]);
    add_line(blk,['IP_Constant',num2str(i),'/1'],['reg_ip',num2str(i),'/1']);
    add_line(blk,['reg_ip',num2str(i),'/1'],['IPMux_Odd/',num2str(ceil(i/2)+1)]);
    
    %add port registers
    reuse_block(blk,['reg_port',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+ceil(i/2)*45+60*numcomputers 185 75+ceil(i/2)*45+60*numcomputers]);
    reuse_block(blk,['Port_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25 45+ceil(i/2)*45+60*numcomputers 55 75+ceil(i/2)*45+60*numcomputers]);
    add_line(blk,['Port_Constant',num2str(i),'/1'],['reg_port',num2str(i),'/1']);
    add_line(blk,['reg_port',num2str(i),'/1'],['PortMux_Odd/',num2str(ceil(i/2)+1)]);
end

start=2;
for i=start:2:2*numcomputers,
    %add ip registers
    reuse_block(blk,['reg_ip',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85+even_x_offset 45+ceil(i/2)*45 185+even_x_offset 75+ceil(i/2)*45]);
    reuse_block(blk,['IP_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', num2str(4000+i),...
        'Position',[25+even_x_offset 45+ceil(i/2)*45 55+even_x_offset 75+ceil(i/2)*45]);
    add_line(blk,['IP_Constant',num2str(i),'/1'],['reg_ip',num2str(i),'/1']);
    add_line(blk,['reg_ip',num2str(i),'/1'],['IPMux_Even/',num2str(ceil(i/2)+1)]);
    
    %add port registers
    reuse_block(blk,['reg_port',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85+even_x_offset 45+ceil(i/2)*45+60*numcomputers 185+even_x_offset 75+ceil(i/2)*45+60*numcomputers]);
    reuse_block(blk,['Port_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25+even_x_offset 45+ceil(i/2)*45+60*numcomputers 55+even_x_offset 75+ceil(i/2)*45+60*numcomputers]);
    add_line(blk,['Port_Constant',num2str(i),'/1'],['reg_port',num2str(i),'/1']);
    add_line(blk,['reg_port',num2str(i),'/1'],['PortMux_Even/',num2str(ceil(i/2)+1)]);
end


clean_blocks(blk);

fmtstr = sprintf('IPs=%d', numcomputers*2);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
