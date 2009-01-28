function[reorderarray] = makereorderarray(numcomputers, numsamples, samplesperpacket)

%numcomputers = 16;                                                  % number of computers to distribute to
%numsamples = 16;                                                    % number of clock cycles it takes for 1 complete spectrum (*2 is number of frequency channels)
samplespercomputer = numsamples/numcomputers;                       % number of samples for each computer
%samplesperpacket = 64;                                              % number of samples for each packet, this number assumes 64 bit data to have 4k packets
spectraperpacket = samplesperpacket/samplespercomputer;             % number of spectra needed to fill a packet

reorderarray = [];
for i=0:numcomputers-1,
    for j=0:spectraperpacket-1,
        reorderarray = horzcat(reorderarray,...
            (j*numsamples+i*samplespercomputer):(j*numsamples+i*samplespercomputer+samplespercomputer-1));
    end
end
    