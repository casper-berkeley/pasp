function gbetest_init()

%% diagram
gbe_rst = xSignal;
gbe_tx_valid = xSignal;
gbe_tx_discard = xSignal;
gbe_rx_ack = xSignal;


const1 = xSignal;
xBlock('Simulink/Sources/Constant',[],{},{const1});


const0 = xSignal;
xBlock('Constant',struct('const',0,'arith_type','Boolean'),{},{const0});

gbe_tx_dest_ip = xSignal;
xBlock(struct('source', 'xps_library/software register', 'name', 'dest_ip'), ...
       struct('io_dir', 'From Processor'), ...
       {const1}, ...
       {gbe_tx_dest_ip});

gbe_tx_dest_port = xSignal;
xBlock(struct('source', 'xps_library/software register', 'name', 'dest_port'), ...
       struct('io_dir', 'From Processor'), ...
       {const1}, ...
       {gbe_tx_dest_port});

reset_counter = xSignal;  
xBlock(struct('source', 'xps_library/software register', 'name', 'reset_counter'), ...
       struct('io_dir', 'From Processor'), ...
       {const1}, ...
       {reset_counter});

packet_size = xSignal;
xBlock(struct('source', 'xps_library/software register', 'name', 'packet_size'), ...
       struct('io_dir', 'From Processor'), ...
       {const1}, ...
       {packet_size});
   

gbe_tx_data = xSignal;
xBlock(struct('source','Counter','name','data_counter'),struct('rst','on'),{reset_counter},{gbe_tx_data});

gbe_tx_eof = xSignal;
xBlock('Relational',[],{gbe_tx_data,packet_size},{gbe_tx_eof});
   
wait_length = xSignal;
xBlock(struct('source', 'xps_library/software register', 'name', 'wait_length'), ...
       struct('io_dir', 'From Processor'), ...
       {const1}, ...
       {wait_length});
   
wait_counter = xSignal;
xBlock(struct('source','Counter','name','wait_counter'),struct('rst','on'),{gbe_tx_eof},{wait_counter});

reset_data_counter = xSignal;
xBlock('Relational',[],{wait_counter,wait_length},{reset_data_counter});



numgbe=1
for i=1:numgbe
    xBlock(struct('source', 'xps_library/ten_GbE', 'name', ['GbE',num2str(i)]), ...
            [], ...
            {const0,gbe_tx_data,gbe_tx_valid,gbe_tx_dest_ip, gbe_tx_dest_port, gbe_tx_eof, const0, const0}, ...
            {});




end

