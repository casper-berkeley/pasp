function [dout, valid, end_of_frame, st, pkt_count, pkt_delay, total_pkt_count] = packetizer(sync, packet_size, packet_wait)

persistent state, state = xl_state(0, {xlUnsigned, 8, 0});
persistent packet_count, packet_count = xl_state(0, {xlUnsigned,64,0});
persistent packetizer_delay, packetizer_delay = xl_state(0,{xlUnsigned,32,0});
persistent total_packet_count, total_packet_count = xl_state(0,{xlUnsigned,64,0});

st=state;
pkt_count=packet_count;
pkt_delay=packetizer_delay;
total_pkt_count=total_packet_count;

% Delay the data by the header size
% dout = dout_delay.back;
% dout_delay.push_front_pop_back(din);
valid = false;
end_of_frame = false;
dout = 0;

% Reset on sync
if sync==true
state=0;
packetizer_delay=0;
packet_count=0;
total_packet_count=0;
end

switch state
% Intitial state: wait for sync or packet delay
case 0
if packetizer_delay==0
state=1;
else
packetizer_delay = packetizer_delay-1;
state=0;
end


% Begin put data into new packet
case 1
if packet_count < packet_size-1
dout = packet_count;
packet_count = packet_count+1;
valid = true;
state=1;
else
dout = total_packet_count;
end_of_frame = true;
valid = true;
state=2;
end

case 2
packetizer_delay = packet_wait-2;
total_packet_count = total_packet_count+1;
state=0;
packet_count=0;

end


end

