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

delete_lines(blk);
reuse_block(blk, 'Count', 'built-in/inport', 'Position', [75    60   105    75], 'Port', '1');
reuse_block(blk, 'Reset', 'built-in/inport', 'Position', [65    25   95    40], 'Port', '2');

reuse_block(blk, 'Counter', 'xbsIndex_r4/Counter', 'n_bits',num2str(ceil(log2(numcomputers))), 'arith_type', 'Unsigned', ...
    'cnt_type', 'Free Running', 'rst', 'On', 'en', 'On', 'Position', [125    47   170    73]);
reuse_block(blk, 'IPMux', 'xbsIndex_r4/Mux','inputs', num2str(numcomputers), 'Latency', '3', 'Position', [260    40   280   40+50*numcomputers]);
reuse_block(blk, 'PortMux', 'xbsIndex_r4/Mux','inputs', num2str(numcomputers), 'Latency', '3', 'Position', [260    40+60*numcomputers   280   40+110*numcomputers]);

reuse_block(blk, 'PortSlice', 'xbsIndex_r4/Slice','nbits','16','mode','Lower Bit Location + Width',...
    'Position', [305 283 335 297]);
reuse_block(blk,'const_0','xbsIndex_r4/Constant',...
	'const','0',...
	'arith_type','Unsigned',...
    'n_bits','32',...
    'bin_pt','0',...
	'ShowName','off',...
	'Position',[290 333 300 347]);
reuse_block(blk, 'Compareto0', 'xbsIndex_r4/Relational',...
    'Position', [305 333 335 347]);
    
reuse_block(blk, 'IP', 'built-in/outport', 'Position', [355   233   385   247], 'Port', '1');
reuse_block(blk, 'Port', 'built-in/outport', 'Position', [355   283   385   297], 'Port', '2');
reuse_block(blk, 'Discard_pkt', 'built-in/outport', 'Position', [355 333 385 347], 'Port','3');

add_line(blk, 'Reset/1', 'Counter/1');
add_line(blk, 'Count/1', 'Counter/2');
add_line(blk, 'Counter/1', 'IPMux/1');
add_line(blk, 'IPMux/1', 'IP/1');
add_line(blk, 'PortMux/1', 'PortSlice/1');
add_line(blk, 'PortSlice/1', 'Port/1');
add_line(blk, 'Counter/1', 'PortMux/1');
add_line(blk, 'const_0/1', 'Compareto0/1');
add_line(blk, 'IPMux/1', 'Compareto0/2');
add_line(blk, 'Compareto0/1', 'Discard_pkt/1');

for i=1:numcomputers,
    %add ip registers
    reuse_block(blk,['reg_ip',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+i*45 185 75+i*45]);
    reuse_block(blk,['IP_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', num2str(4000+i),...
        'Position',[25 45+i*45 55 75+i*45]);
    add_line(blk,['IP_Constant',num2str(i),'/1'],['reg_ip',num2str(i),'/1']);
    add_line(blk,['reg_ip',num2str(i),'/1'],['IPMux/',num2str(i+1)]);
    
    %add port registers
    reuse_block(blk,['reg_port',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+i*45+60*numcomputers 185 75+i*45+60*numcomputers]);
    reuse_block(blk,['Port_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25 45+i*45+60*numcomputers 55 75+i*45+60*numcomputers]);
    add_line(blk,['Port_Constant',num2str(i),'/1'],['reg_port',num2str(i),'/1']);
    add_line(blk,['reg_port',num2str(i),'/1'],['PortMux/',num2str(i+1)]);
end


clean_blocks(blk);

fmtstr = sprintf('IPs=%d', numcomputers);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
