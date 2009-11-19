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

if numchannels>=8,
    even_x_offset=500;

    delete_lines(blk);
    reuse_block(blk, 'Reset', 'built-in/inport', 'Position', [65    25   95    40], 'Port', '1');

    reuse_block(blk, 'Counter', 'xbsIndex_r4/Counter', 'n_bits',num2str(log2(numchannels)), 'arith_type', 'Unsigned', ...
        'cnt_type', 'Free Running', 'rst', 'On', 'en', 'Off', 'Position', [125    47   170    73]);

    reuse_block(blk, 'data_in', 'xbsIndex_r4/Constant', 'n_bits', '32', 'arith_type', 'Unsigned', ...
        'explicit_period', 'On', 'Position', [125    107   170    133]);

    reuse_block(blk, 'we', 'xbsIndex_r4/Constant', 'arith_type', 'Boolean', 'const', '0', ...
        'explicit_period', 'On', 'Position', [125    167   170    193]);

    reuse_block(blk,'pol0_even_bram','xps_library/Shared BRAM',...
        'arith_type','Unsigned',...
        'addr_width','11',...
        'init_vals',['1:',num2str(numchannels)],...
        'Position',[355    55   455    85]);
    reuse_block(blk,'pol0_odd_bram','xps_library/Shared BRAM',...
        'arith_type','Unsigned',...
        'addr_width','11',...
        'init_vals',['1:',num2str(numchannels)],...
        'Position',[355   140   455   170]);
    reuse_block(blk,'pol1_even_bram','xps_library/Shared BRAM',...
        'arith_type','Unsigned',...
        'addr_width','11',...
        'init_vals',['1:',num2str(numchannels)],...
        'Position',[355   225   455   255]);
    reuse_block(blk,'pol1_odd_bram','xps_library/Shared BRAM',...
        'arith_type','Unsigned',...
        'addr_width','11',...
        'init_vals',['1:',num2str(numchannels)],...
        'Position',[355   310   455   340]);

    reuse_block(blk, 'Pol0_Odd', 'built-in/outport', 'Position', [505   148   535   162], 'Port', '1');
    reuse_block(blk, 'Pol1_Odd', 'built-in/outport', 'Position', [505   318   535   332], 'Port', '3');
    reuse_block(blk, 'Pol0_Even', 'built-in/outport', 'Position', [505    63   535    77], 'Port', '2');
    reuse_block(blk, 'Pol1_Even', 'built-in/outport', 'Position', [505   233   535   247], 'Port', '4');

    add_line(blk, 'Reset/1', 'Counter/1');

    add_line(blk, 'Counter/1', 'pol0_odd_bram/1');
    add_line(blk, 'data_in/1', 'pol0_odd_bram/2');
    add_line(blk, 'we/1', 'pol0_odd_bram/3');
    add_line(blk, 'pol0_odd_bram/1', 'Pol0_Odd/1');

    add_line(blk, 'Counter/1', 'pol0_even_bram/1');
    add_line(blk, 'data_in/1', 'pol0_even_bram/2');
    add_line(blk, 'we/1', 'pol0_even_bram/3');
    add_line(blk, 'pol0_even_bram/1', 'Pol0_Even/1');

    add_line(blk, 'Counter/1', 'pol1_odd_bram/1');
    add_line(blk, 'data_in/1', 'pol1_odd_bram/2');
    add_line(blk, 'we/1', 'pol1_odd_bram/3');
    add_line(blk, 'pol1_odd_bram/1', 'Pol1_Odd/1');

    add_line(blk, 'Counter/1', 'pol1_even_bram/1');
    add_line(blk, 'data_in/1', 'pol1_even_bram/2');
    add_line(blk, 'we/1', 'pol1_even_bram/3');
    add_line(blk, 'pol1_even_bram/1', 'Pol1_Even/1');
else,
    even_x_offset=500;

    delete_lines(blk);
    reuse_block(blk, 'Reset', 'built-in/inport', 'Position', [65    25   95    40], 'Port', '1');

    reuse_block(blk, 'Counter', 'xbsIndex_r4/Counter', 'n_bits',num2str(log2(numchannels)), 'arith_type', 'Unsigned', ...
        'cnt_type', 'Free Running', 'rst', 'On', 'en', 'Off', 'Position', [125    47   170    73]);
    reuse_block(blk, 'IPMux_Odd', 'xbsIndex_r4/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260    40   280   40+50*numchannels]);
    reuse_block(blk, 'PortMux_Odd', 'xbsIndex_r4/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260    40+60*numchannels   280   40+110*numchannels]);

    reuse_block(blk, 'Delay_Even', 'xbsIndex_r4/Delay', 'latency', '4', 'Position', [410    43   435    67]);
    reuse_block(blk, 'IPMux_Even', 'xbsIndex_r4/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260+even_x_offset    40   280+even_x_offset   40+50*numchannels]);
    reuse_block(blk, 'PortMux_Even', 'xbsIndex_r4/Mux','inputs', num2str(numchannels), 'Latency', '3', 'Position', [260+even_x_offset    40+60*numchannels   280+even_x_offset   40+110*numchannels]);

    reuse_block(blk, 'Pol0_Odd', 'built-in/outport', 'Position', [305   233   335   247], 'Port', '1');
    reuse_block(blk, 'Pol0_Even', 'built-in/outport', 'Position', [305+even_x_offset   233   335+even_x_offset   247], 'Port', '2');
    reuse_block(blk, 'Pol1_Odd', 'built-in/outport', 'Position', [305   283   335   297], 'Port', '3');
    reuse_block(blk, 'Pol1_Even', 'built-in/outport', 'Position', [305+even_x_offset   283   335+even_x_offset   297], 'Port', '4');

    add_line(blk, 'Reset/1', 'Counter/1');
    add_line(blk, 'Counter/1', 'IPMux_Odd/1');
    add_line(blk, 'IPMux_Odd/1', 'Pol0_Odd/1');
    add_line(blk, 'PortMux_Odd/1', 'Pol1_Odd/1');
    add_line(blk, 'Counter/1', 'PortMux_Odd/1');

    add_line(blk, 'Counter/1', 'IPMux_Even/1');
    add_line(blk, 'IPMux_Even/1', 'Pol0_Even/1');
    add_line(blk, 'PortMux_Even/1', 'Pol1_Even/1');
    add_line(blk, 'Counter/1', 'PortMux_Even/1');

    start=1;
    for i=start:2:2*numchannels,
        %add ip registers
        reuse_block(blk,['scale_pol0_ch',num2str(i)],'xps_library/software register',...
            'io_dir','From Processor',...
            'Position',[85 45+ceil(i/2)*45 185 75+ceil(i/2)*45]);
        reuse_block(blk,['IP_Constant',num2str(i)],'simulink/Sources/Constant',...
            'Value', num2str(4000+i),...
            'Position',[25 45+ceil(i/2)*45 55 75+ceil(i/2)*45]);
        add_line(blk,['IP_Constant',num2str(i),'/1'],['scale_pol0_ch',num2str(i),'/1']);
        add_line(blk,['scale_pol0_ch',num2str(i),'/1'],['IPMux_Odd/',num2str(ceil(i/2)+1)]);

        %add port registers
        reuse_block(blk,['scale_pol1_ch',num2str(i)],'xps_library/software register',...
            'io_dir','From Processor',...
            'Position',[85 45+ceil(i/2)*45+60*numchannels 185 75+ceil(i/2)*45+60*numchannels]);
        reuse_block(blk,['Port_Constant',num2str(i)],'simulink/Sources/Constant',...
            'Value', '4000',...
            'Position',[25 45+ceil(i/2)*45+60*numchannels 55 75+ceil(i/2)*45+60*numchannels]);
        add_line(blk,['Port_Constant',num2str(i),'/1'],['scale_pol1_ch',num2str(i),'/1']);
        add_line(blk,['scale_pol1_ch',num2str(i),'/1'],['PortMux_Odd/',num2str(ceil(i/2)+1)]);
    end

    start=2;
    for i=start:2:2*numchannels,
        %add ip registers
        reuse_block(blk,['scale_pol0_ch',num2str(i)],'xps_library/software register',...
            'io_dir','From Processor',...
            'Position',[85+even_x_offset 45+ceil(i/2)*45 185+even_x_offset 75+ceil(i/2)*45]);
        reuse_block(blk,['IP_Constant',num2str(i)],'simulink/Sources/Constant',...
            'Value', num2str(4000+i),...
            'Position',[25+even_x_offset 45+ceil(i/2)*45 55+even_x_offset 75+ceil(i/2)*45]);
        add_line(blk,['IP_Constant',num2str(i),'/1'],['scale_pol0_ch',num2str(i),'/1']);
        add_line(blk,['scale_pol0_ch',num2str(i),'/1'],['IPMux_Even/',num2str(ceil(i/2)+1)]);

        %add port registers
        reuse_block(blk,['scale_pol1_ch',num2str(i)],'xps_library/software register',...
            'io_dir','From Processor',...
            'Position',[85+even_x_offset 45+ceil(i/2)*45+60*numchannels 185+even_x_offset 75+ceil(i/2)*45+60*numchannels]);
        reuse_block(blk,['Port_Constant',num2str(i)],'simulink/Sources/Constant',...
            'Value', '4000',...
            'Position',[25+even_x_offset 45+ceil(i/2)*45+60*numchannels 55+even_x_offset 75+ceil(i/2)*45+60*numchannels]);
        add_line(blk,['Port_Constant',num2str(i),'/1'],['scale_pol1_ch',num2str(i),'/1']);
        add_line(blk,['scale_pol1_ch',num2str(i),'/1'],['PortMux_Even/',num2str(ceil(i/2)+1)]);
    end  
    
end





clean_blocks(blk);

fmtstr = sprintf('Channels=%d', numchannels*2);
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
