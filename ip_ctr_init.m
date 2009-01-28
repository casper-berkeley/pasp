function ip_ctr_init(blk, varargin)
% Initialize and configure the ip counter block.
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
defaults = {'numcomputers', 8};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'ip_ctr');
munge_block(blk, varargin{:});

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
odd_channels = get_var('odd_channels', 'defaults', defaults, varargin{:});

if(odd_channels),
    start=1;
else,
    start=2;
end

delete_lines(blk);
reuse_block(blk, 'Count', 'built-in/inport', 'Position', [95    25   125    40], 'Port', '1');
reuse_block(blk, 'Reset', 'built-in/inport', 'Position', [95    25   125    40], 'Port', '2');
reuse_block(blk, 'Counter1', 'xbsIndex_r3/Counter', 'n_bits',num2str(log2(numcomputers)), 'Position', [125    47   170    73]);
reuse_block(blk, 'IPMux', 'xbsIndex_r3/Mux','inputs', num2str(numcomputers), 'Position', [260    40   280   40+50*numcomputers]);
reuse_block(blk, 'PortMux', 'xbsIndex_r3/Mux','inputs', num2str(numcomputers), 'Position', [260    40+60*numcomputers   280   40+110*numcomputers]);
reuse_block(blk, 'IP', 'built-in/outport', 'Position', [305   233   335   247], 'Port', '1');
reuse_block(blk, 'Port', 'built-in/outport', 'Position', [305   283   335   297], 'Port', '2');

add_line(blk, 'Reset/1', 'Counter1/1');
add_line(blk, 'Count/1', 'Counter1/2');
add_line(blk, 'Counter1/1', 'IPMux/1');
add_line(blk, 'IPMux/1', 'IP/1');
add_line(blk, 'PortMux/1', 'Port/1');
add_line(blk, 'Counter1/1', 'PortMux/1');

for i=start:2:2*numcomputers,
    %add ip registers
    reuse_block(blk,['reg_ip',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+ceil(i/2)*45 185 75+ceil(i/2)*45]);
    reuse_block(blk,['IP_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25 45+ceil(i/2)*45 55 75+ceil(i/2)*45]);
    add_line(blk,['IP_Constant',num2str(i),'/1'],['reg_ip',num2str(i),'/1']);
    add_line(blk,['reg_ip',num2str(i),'/1'],['IPMux/',num2str(ceil(i/2)+1)]);
    
    %add port registers
    reuse_block(blk,['reg_port',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+ceil(i/2)*45+60*numcomputers 185 75+ceil(i/2)*45+60*numcomputers]);
    reuse_block(blk,['Port_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25 45+ceil(i/2)*45+60*numcomputers 55 75+ceil(i/2)*45+60*numcomputers]);
    add_line(blk,['Port_Constant',num2str(i),'/1'],['reg_port',num2str(i),'/1']);
    add_line(blk,['reg_port',num2str(i),'/1'],['PortMux/',num2str(ceil(i/2)+1)]);
end


clean_blocks(blk);

fmtstr = sprintf('IPs=%d', numcomputers);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
