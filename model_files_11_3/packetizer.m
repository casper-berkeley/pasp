function [dout, valid, end_of_frame, st, cidout] = packetizer(sync, din, sys_counter, packet_size, numcomputers, numtengbe, tengbe_id)

persistent state, state = xl_state(0, {xlUnsigned, 8, 0});
persistent channel_id, channel_id = xl_state(tengbe_id, {xlUnsigned,64,0});
persistent packet_count, packet_count = xl_state(0, {xlUnsigned,64,0});
persistent dout_delay, dout_delay = xl_state(zeros(1,2), {xlUnsigned,64,0}, 2);
persistent packetizer_delay, packetizer_delay = xl_state(packet_size*tengbe_id,{xlUnsigned,64,0});

st=state;

% Delay the data by the header size
dout = dout_delay.back;
dout_delay.push_front_pop_back(din);
valid = false;
end_of_frame = false;
cidout = channel_id;

% Reset on sync
if sync==true
    state=0;
    packetizer_delay=packet_size*tengbe_id;
    channel_id=tengbe_id;
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
   
    case 1
        dout = sys_counter;
        valid = true;
        state = 2;
        
    case 2
        dout=channel_id;
        channel_id=channel_id+numtengbe;
        if channel_id >= numcomputers
            channel_id=tengbe_id;
        end
        
        valid = true;
        
        state = 3;
        
        packet_count = 0;
    
    % Begin put data into new packet
    case 3
        if packet_count < packet_size-1
            packet_count = packet_count+1;
            valid = true;
            state=3;
        else
            end_of_frame = true;
            valid = true;
            state=4;
        end
    case 4
        packetizer_delay = (numtengbe-1)*packet_size-1-2;
        packetizer_delay = packetizer_delay-1;
        state=0;
        
end
    
    
end

