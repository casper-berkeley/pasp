function dist_gbe_init(blk, varargin)
% Initialize and configure the packet distributer block.
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
defaults = {'numcomputers', 16, 'numsamples', 16, 'samplesperpacket', 64, 'numtengbe',2};
if same_state(blk, 'defaults', defaults, varargin{:}), return, end
check_mask_type(blk, 'dist_gbe');
munge_block(blk, varargin{:});
delete_lines(blk);

numcomputers = get_var('numcomputers', 'defaults', defaults, varargin{:});
numsamples = get_var('numsamples', 'defaults', defaults, varargin{:});
samplesperpacket = get_var('samplesperpacket', 'defaults', defaults, varargin{:});
numtengbe = get_var('numtengbe', 'defaults', defaults, varargin{:});

load_system('pasp_lib.mdl');

reuse_block(blk, 'sync', 'built-in/inport', 'Position', [75   402   105   418], 'Port', '1');
reuse_block(blk, 'data_in', 'built-in/inport', 'Position', [75   442   105   458], 'Port', '2');
reuse_block(blk, 'sys_ctr', 'built-in/inport', 'Position', [75   492   105   508], 'Port', '3');

reuse_block(blk,'reorder_en','xbsIndex_r4/Constant',...
    'const','1',...
    'arith_type','Boolean',...
    'explicit_period','on',...
    'ShowName','off',...
    'Position',[140   422   170   438]);

% update the reorder ordering
reuse_block(blk,'reorder','casper_library/Reorder/reorder',...
    'n_inputs','1',...
    'map',['makereorderarray(',num2str(numcomputers),', ', num2str(numsamples), ', ', num2str(samplesperpacket), ')'],...
    'Position',[195   404   290   476]);

add_line(blk,'sync/1','reorder/1');
add_line(blk,'reorder_en/1','reorder/2');
add_line(blk,'data_in/1','reorder/3');

reuse_block(blk,'ip_ctr','pasp_lib/ip_ctr',...
    'numcomputers',num2str(numcomputers),...
    'Position',[1440         105        1565         245]);

% or together send packet signals to increment ip counter
reuse_block(blk,'next_packet','xbsIndex_r4/Logical',...
    'logical_function','OR',...
    'inputs',num2str(numtengbe),...
    'Position',[1340         109        1395         171]);
add_line(blk,'next_packet/1','ip_ctr/1');

% reset the ip counter on sync
add_line(blk,'reorder/1','ip_ctr/2');

for i=0:numtengbe-1,
    
    reuse_block(blk,['packetizer',num2str(i)],'pasp_lib/packetizer',...
        'Position',[700 300*i+370 1150 300*i+600]);
    
    set_param([blk,'/packetizer',num2str(i)],...
        'defparams',['{''packet_size'',',num2str(samplesperpacket),...
        ',''numcomputers'',',num2str(numcomputers),...
        ',''numtengbe'',',num2str(numtengbe),...
        ',''tengbe_id'',',num2str(i),'}']);
    
    reuse_block(blk,['delay_dout_',num2str(i)],'xbsIndex_r4/Delay',...
        'Position',[1225         300*i+386        1285         300*i+434]);
    reuse_block(blk,['delay_valid_',num2str(i)],'xbsIndex_r4/Delay',...
        'Position',[1225         300*i+461        1285         300*i+509]);
    reuse_block(blk,['delay_eof_',num2str(i)],'xbsIndex_r4/Delay',...
        'Position',[1225         300*i+536        1285         300*i+584]);
    
    reuse_block(blk,['ten_GbE',num2str(i),'_rst'],'xbsIndex_r4/Constant',...
        'const','0',...
        'arith_type','Boolean',...
        'ShowName','off',...
        'Position',[1695 300*i+366 1740 300*i+394]);
    
    reuse_block(blk,['discard_and_',num2str(i)],'xbsIndex_r4/Logical',...
        'logical_function','AND',...
        'Position',[1695 300*i+450 1740 300*i+490]);
    
    reuse_block(blk,['ten_GbE',num2str(i),'_ack'],'xbsIndex_r4/Constant',...
        'const','0',...
        'arith_type','Boolean',...
        'ShowName','off',...
        'Position',[1695 300*i+576 1740 300*i+604]);
    
    reuse_block(blk,['ten_GbE',num2str(i)],'xps_library/ten_GbE',...
        'port',['ROACH:',num2str(i)],...
        'Position',[1800 300*i+370 2100 300*i+600]);
    
    add_line(blk,['packetizer',num2str(i),'/1'],['delay_dout_',num2str(i),'/1']);
    add_line(blk,['packetizer',num2str(i),'/2'],['delay_valid_',num2str(i),'/1']);
    add_line(blk,['packetizer',num2str(i),'/3'],['delay_eof_',num2str(i),'/1']);
    
    add_line(blk,'reorder/1',['packetizer',num2str(i),'/1']);
    add_line(blk,'reorder/3',['packetizer',num2str(i),'/2']);
    add_line(blk,'sys_ctr/1',['packetizer',num2str(i),'/3']);
    
    add_line(blk,['delay_eof_',num2str(i),'/1'],['next_packet/',num2str(i+1)]);
    
    add_line(blk,'ip_ctr/3',['discard_and_',num2str(i),'/1']);
    add_line(blk,['delay_eof_',num2str(i),'/1'],['discard_and_',num2str(i),'/2']);
    
    add_line(blk,['ten_GbE',num2str(i),'_rst/1'],['ten_GbE',num2str(i),'/1']);
    add_line(blk,['delay_dout_',num2str(i),'/1'],['ten_GbE',num2str(i),'/2']);
    add_line(blk,['delay_valid_',num2str(i),'/1'],['ten_GbE',num2str(i),'/3']);
    add_line(blk,'ip_ctr/1',['ten_GbE',num2str(i),'/4']);
    add_line(blk,'ip_ctr/2',['ten_GbE',num2str(i),'/5']);
    add_line(blk,['delay_eof_',num2str(i),'/1'],['ten_GbE',num2str(i),'/6']);
    add_line(blk,['discard_and_',num2str(i),'/1'],['ten_GbE',num2str(i),'/7']);
    add_line(blk,['ten_GbE',num2str(i),'_ack/1'],['ten_GbE',num2str(i),'/8']);
    
end



clean_blocks(blk);

fmtstr = sprintf('IPs=%d\n%s', numcomputers, get_param([blk,'/reorder'], 'AttributesFormatString'));
set_param(blk, 'AttributesFormatString', fmtstr);
save_state(blk, 'defaults', defaults, varargin{:});
