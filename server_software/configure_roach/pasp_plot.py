import pylab

# plot fft data until interrupt received   
def plot_fft_brams(new_pasp):
    run = True
    
    pylab.ion()
    pylab.yscale('log')
    pylab.ylim(0,10000000)
    
    fftscope_power = new_pasp.get_fft_brams_power()
    fftscope_power_line=pylab.bar(range(0,new_pasp.numchannels),fftscope_power)
    

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