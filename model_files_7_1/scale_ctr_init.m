function scale_ctr_init(blk, varargin)
% Initialize and configure the ip counter block.
%
% reorder_init(blk, varargin)
%
% blk = The block to be initialize.
% varargin = {'varname', 'value', ...} pairs
% 
% Valid varnames for this block are:


% Declare any default values for arguments you might like.
defaults = {'numchannels', 8};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'scale_ctr');
munge_block(blk, varargin{:});

numchannels = get_var('numchannels', 'defaults', defaults, varargin{:});
% calculate the number of odd and number of even ips
numchannels=numchannels/2;

even_x_offset=500;

delete_lines(blk);
reuse_block(blk, 'Reset', 'built-in/inport', 'Position', [65    25   95    40], 'Port', '1');

reuse_block(blk, 'Counter', 'xbsIndex_r3/Counter', 'n_bits',num2str(log2(numchannels)), 'arith_type', 'Unsigned', ...
    'cnt_type', 'Free Running', 'rst', 'On', 'en', 'Off', 'Position', [125    47   170    73]);
reuse_block(blk, 'Pol0_Mux_Odd', 'xbsIndex_r3/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260    40   280   40+50*numchannels]);
reuse_block(blk, 'Pol1_Mux_Odd', 'xbsIndex_r3/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260    40+60*numchannels   280   40+110*numchannels]);

reuse_block(blk, 'Pol0_Mux_Even', 'xbsIndex_r3/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260+even_x_offset    40   280+even_x_offset   40+50*numchannels]);
reuse_block(blk, 'Pol1_Mux_Even', 'xbsIndex_r3/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260+even_x_offset    40+60*numchannels   280+even_x_offset   40+110*numchannels]);

reuse_block(blk, 'Pol0_Odd', 'built-in/outport', 'Position', [305   233   335   247], 'Port', '1');
reuse_block(blk, 'Pol1_Odd', 'built-in/outport', 'Position', [305   283   335   297], 'Port', '3');
reuse_block(blk, 'Pol0_Even', 'built-in/outport', 'Position', [305+even_x_offset   233   335+even_x_offset   247], 'Port', '2');
reuse_block(blk, 'Pol1_Even', 'built-in/outport', 'Position', [305+even_x_offset   283   335+even_x_offset   297], 'Port', '4');

add_line(blk, 'Reset/1', 'Counter/1');
add_line(blk, 'Counter/1', 'Pol0_Mux_Odd/1');
add_line(blk, 'Pol0_Mux_Odd/1', 'Pol0_Odd/1');
add_line(blk, 'Pol1_Mux_Odd/1', 'Pol1_Odd/1');
add_line(blk, 'Counter/1', 'Pol1_Mux_Odd/1');

add_line(blk, 'Counter/1', 'Pol0_Mux_Even/1');
add_line(blk, 'Pol0_Mux_Even/1', 'Pol0_Even/1');
add_line(blk, 'Pol1_Mux_Even/1', 'Pol1_Even/1');
add_line(blk, 'Counter/1', 'Pol1_Mux_Even/1');

start=1;
for i=start:2:2*numchannels,
    %add ip registers
    reuse_block(blk,['scale_pol0_',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+ceil(i/2)*45 185 75+ceil(i/2)*45]);
    reuse_block(blk,['Pol0_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', num2str(2^12),...
        'Position',[25 45+ceil(i/2)*45 55 75+ceil(i/2)*45]);
    add_line(blk,['Pol0_Constant',num2str(i),'/1'],['scale_pol0_',num2str(i),'/1']);
    add_line(blk,['scale_pol0_',num2str(i),'/1'],['Pol0_Mux_Odd/',num2str(ceil(i/2)+1)]);
    
    %add port registers
    reuse_block(blk,['scale_pol1_',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85 45+ceil(i/2)*45+60*numchannels 185 75+ceil(i/2)*45+60*numchannels]);
    reuse_block(blk,['Pol1_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25 45+ceil(i/2)*45+60*numchannels 55 75+ceil(i/2)*45+60*numchannels]);
    add_line(blk,['Pol1_Constant',num2str(i),'/1'],['scale_pol1_',num2str(i),'/1']);
    add_line(blk,['scale_pol1_',num2str(i),'/1'],['Pol1_Mux_Odd/',num2str(ceil(i/2)+1)]);
end

start=2;
for i=start:2:2*numchannels,
    %add ip registers
    reuse_block(blk,['scale_pol0_',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85+even_x_offset 45+ceil(i/2)*45 185+even_x_offset 75+ceil(i/2)*45]);
    reuse_block(blk,['Pol0_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', num2str(2^12),...
        'Position',[25+even_x_offset 45+ceil(i/2)*45 55+even_x_offset 75+ceil(i/2)*45]);
    add_line(blk,['Pol0_Constant',num2str(i),'/1'],['scale_pol0_',num2str(i),'/1']);
    add_line(blk,['scale_pol0_',num2str(i),'/1'],['Pol0_Mux_Even/',num2str(ceil(i/2)+1)]);
    
    %add port registers
    reuse_block(blk,['scale_pol1_',num2str(i)],'xps_library/software register',...
        'io_dir','From Processor',...
        'Position',[85+even_x_offset 45+ceil(i/2)*45+60*numchannels 185+even_x_offset 75+ceil(i/2)*45+60*numchannels]);
    reuse_block(blk,['Pol1_Constant',num2str(i)],'simulink/Sources/Constant',...
        'Value', '4000',...
        'Position',[25+even_x_offset 45+ceil(i/2)*45+60*numchannels 55+even_x_offset 75+ceil(i/2)*45+60*numchannels]);
    add_line(blk,['Pol1_Constant',num2str(i),'/1'],['scale_pol1_',num2str(i),'/1']);
    add_line(blk,['scale_pol1_',num2str(i),'/1'],['Pol1_Mux_Even/',num2str(ceil(i/2)+1)]);
end


clean_blocks(blk);

fmtstr = sprintf('Channels=%d', numchannels*2);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
