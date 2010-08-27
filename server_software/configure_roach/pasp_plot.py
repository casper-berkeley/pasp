import pylab

# plot fft data until interrupt received   
def plot_fft_brams(new_pasp):

    run = True
    
    pylab.ion()
    pylab.cla()
    pylab.yscale('log')
    
    
    
    # read in initial data from the fft brams
    fftscope_power = new_pasp.get_fft_brams_power()
    # set up bars for each pasp channel
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
    
    # after receiving an interrupt wait before closing the plot
    raw_input('Press enter to quit: ')
            
    pylab.cla()    
    
    

# plot adc data
def plot_adc_brams(new_pasp):

    run = True
    
    pylab.ion()
    #pylab.yscale('log')

    # read in the adc data
    adcscope_pol1, adcscope_pol2 = new_pasp.get_adc_brams(100)
    print adcscope_pol1
    pylab.cla()
    adcscope_power_line, = pylab.plot(adcscope_pol1)
    
    #while(run):
    '''
    for i in range(1,10):
        try:
            fftscope_power = get_fft_brams_power()
            fftscope_power_line.set_ydata(fftscope_power)
            pylab.draw()

        except KeyboardInterrupt:
            run = False
    '''
    
    raw_input('Press enter to quit: ')
            
    pylab.cla()  
    