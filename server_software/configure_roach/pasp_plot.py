
# plot fft data until interrupt received   
def plot_fft_brams(new_pasp):
    import pylab
    run = True
    
    pylab.ion()
    pylab.yscale('log')
    
    
    fftscope_power = new_pasp.get_fft_brams_power()
    fftscope_power_line=pylab.bar(range(0,new_pasp.numchannels),fftscope_power)
    
    pylab.ylim(1,1000000)
    
    # plot forever
    #for i in range(1,10):
    while(run):
        try:
            fftscope_power = new_pasp.get_fft_brams_power()
            
            # update the rectangles 
            for j in range(0,new_pasp.numchannels):
                fftscope_power_line[j].set_height(fftscope_power[j])
            pylab.draw()
        except KeyboardInterrupt:
            run = False
        

    """
    fftscope_power_line, = pylab.plot(fftscope_power,'bo')
    
    #while(run):
    
    for i in range(1,10):
        try:
            fftscope_power = get_fft_brams_power()
            fftscope_power_line.set_ydata(fftscope_power)
            pylab.draw()

        except KeyboardInterrupt:
            run = False
    """
            
    pylab.cla()
    
    raw_input('Press enter to quit: ')