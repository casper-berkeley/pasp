function [dout, valid, end_of_frame, st, cidout] = packetizer(sync, din, sys_counter, packet_size, numcomputers, numtengbe, tengbe_id)

persistent state, state = xl_state(0, {xlUnsigned, 2, 0});
persistent channel_id, channel_id = xl_state(0, {xlUnsigned,16,0});
persistent packet_count, packet_count = xl_state(0, {xlUnsigned,16,0});
persistent dout_delay, dout_delay = xl_state(zeros(1,2), {xlUnsigned,64,0}, 2);
persistent packetizer_delay, packetizer_delay = xl_state(0,{xlUnsigned,16,0});

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
% Otherwise go through FSM
else,
    switch state
        % Intitial state: wait for sync or packet delay
        case 0
            % add syscounter to the packet
            if packetizer_delay==0
                dout = sys_counter;
                valid = true;
                state = 1;
                packetizer_delay = (numtengbe-1)*packet_size;
            else
                packetizer_delay = packetizer_delay-1;
                state=0;
            end

        % add channel_id to the packet
        case 1
            dout=channel_id;
            valid = true;
            state = 2;
            packet_count = 0;
        
        % Put data into new packet
        case 2
            % add packet_size-1 data elements into the packet
            if packet_count < packet_size-1
                packet_count = packet_count+1;
                valid = true;
                state=2;
            % add the final data element and pulse eof
            else
                end_of_frame = true;
                valid = true;
                state=3;
                channel_id=channel_id+numtengbe;
            end
        
        case 3
            packetizer_delay = packetizer_delay-1;
            % originally in state 2 but trying to add some latency for timing
            if channel_id >= numcomputers
                channel_id=tengbe_id;
            end
            state=0;
    end
end    
    
end

