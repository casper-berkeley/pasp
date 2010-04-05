#!/usr/bin/env python2.6

import corr, time, struct, sys, logging, socket, re


# fft configuration
# shifting schedule
fft_shift=0xfffffffe

# fft scaling configuration
# interpreted as 18.12 ufix
fft_coeffs=[4096,4096,4096,4096,4096,4096,4096,4096]    # no scaling (multiply by 1)


# bit select configuration from 8 bits from scaled 18 bit fft data
# 0-select bottom 8 bits
# 1-
# 2-
# 3-select top 8 bits
bitselect_pol1=0 # select bottom bits
bitselect_pol2=0 # select bottom bits

# ip table configuration
ip_table=['10.0.0.14','10.0.0.14','10.0.0.14','10.0.0.14']
port_table=[6000,6000,6000,6000]





# default boffile
defaultbof='pasp_4i16c1024s4g.bof'

ip_reg_base='pasp_dist_gbe_ip_ctr_reg_ip'
port_reg_base='pasp_dist_gbe_ip_ctr_reg_port'


gbe_base='pasp_dist_gbe_ten_GbE'

mac_base=(2<<40) + (2<<32)
fabric_ip=struct.unpack('!L',socket.inet_aton('10.0.0.30'))[0] #convert ip to long
fabric_port=6000


katcp_port=7147


def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    #exit()

def exit_clean():
    #print 'NO FAILURES'
    try:
        for f in fpgas: f.stop()
    except: pass
    #exit()
    
# functions for dumping data    
def dump_adc_brams():
    #trigger the adcscope
    fpga.write_int('pasp_reg_adcscope_trigger',1)
    time.sleep(1)
    fpga.write_int('pasp_reg_adcscope_trigger',0)
    
    #read the bram data
    adcscope1 = struct.unpack('>2048b',fpga.read('pasp_bram_adcscope_1',2048))
    adcscope2 = struct.unpack('>2048b',fpga.read('pasp_bram_adcscope_2',2048))
    print adcscope1
    print adcscope2


def dump_fft_brams():
    fftscope1 = struct.unpack('>2048l',fpga.read('pasp_scope_output1_bram',2048*4))
    fftscope2 = struct.unpack('>2048l',fpga.read('pasp_scope_output2_bram',2048*4))
    fftscope3 = struct.unpack('>2048l',fpga.read('pasp_scope_output3_bram',2048*4))
    fftscope4 = struct.unpack('>2048l',fpga.read('pasp_scope_output4_bram',2048*4))
    
    print fftscope1
    print fftscope2
    print fftscope3
    print fftscope4
   
def configure_parameters():
    # initialize the fpga sync
    logger.debug('Configuring a sync period of %d'%(sync_period))
    fpga.write_int('pasp_reg_sync_period',sync_period)
    
    # initialize the fft shift schedule
    fpga.write_int('pasp_reg_fft_shift',fft_shift)
    #fpga.write_int('pasp_reg_fft_shift',0x0)
    
    # initialize the scaling parameters
    fft_coeffs_string = struct.pack('>8L',*fft_coeffs)
    fpga.write('pasp_scale_ctr_pol0_even_bram',fft_coeffs_string)
    fpga.write('pasp_scale_ctr_pol0_odd_bram',fft_coeffs_string)
    fpga.write('pasp_scale_ctr_pol1_even_bram',fft_coeffs_string)
    fpga.write('pasp_scale_ctr_pol1_odd_bram',fft_coeffs_string)

    # initialize the scaling bit selection
    fpga.write_int('pasp_reg_output_bitselect_pol1',bitselect_pol1)
    fpga.write_int('pasp_reg_output_bitselect_pol2',bitselect_pol2)

      
def init_10gbe_blocks():
    # initialize the 10gbe ports
    for i in range(0,numtengbe):
        logger.debug('Initializing '+gbe_base+str(i))
        fpga.tap_start('gbe'+str(i),gbe_base+str(i),mac_base+fabric_ip+i,fabric_ip+i,fabric_port)
    
def init_ip_table():
    # initialize the ip table
    logger.debug('Initializing ip table')
    for i in range(0,numips):
        fpga.write_int(ip_reg_base+str(i),struct.unpack('!L',socket.inet_aton(ip_table[i]))[0])
        fpga.write_int(port_reg_base+str(i),port_table[i])

        
def clear_ip_table():
    # zero the ip table
    print 'Zeroing ip table'
    for i in range(0,numips):
        fpga.write_int(ip_reg_base+str(i),0)
    
    
if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('pasp_init.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.set_defaults(interactive_mode=False)
    p.set_defaults(boffile=defaultbof)
    p.add_option('-i','--interactive', help='select boffile interactively', action='store_true', dest='interactive_mode')
    p.add_option('-f','--file', help='use specified boffile', action='store', type='string', dest='boffile')
 
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'No ROACH board specified. Defaulting to ROACH01'
        roach = 'roach01'
    else:
        roach = args[0]

try:
    loggers = []
    lh=corr.log_handlers.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s on port %i... '%(roach,katcp_port))
    fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout=10,logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s on port %i.\n'%(roach,katcp_port)
        exit_fail()
        
    # list boffiles and select interactively
    if opts.interactive_mode == True:
        print 'Available pasp boffiles:'  
        for testboffile in fpga.listbof():
            if 'pasp' in testboffile:
                print testboffile
        opts.boffile=raw_input('Select a boffile: ')
    
    
    
    # extract build configuration info from boffile name
    m=re.search('pasp_([\d]+)i([\d]+)c([\d]+)s([\d]+)g',opts.boffile)
    
    # get pasp build configuration from boffile name
    numips=int(m.group(1))
    numchannels=int(m.group(2))
    packetsize=int(m.group(3))
    numtengbe=int(m.group(4))
    reorder_order=3
    
    logger.debug('Got numips %d numchannels %d packetsize %d numtengbe %d'%(numips,numchannels,packetsize,numtengbe))

    # calculate the sync period
    # LCM(reorder orders)*(FFTSize/simultaneousinputs)*numtengbe*packetsize
    sync_period=reorder_order*numchannels*numtengbe*packetsize
            
    # program the fpga with selected boffile
    logger.debug('Clearing the fpga')       
    fpga.progdev('')
    time.sleep(1)
    logger.debug('Programming the fpga with %s'%(opts.boffile))   
    print 'Programming the fpga with %s'%(opts.boffile)  
    fpga.progdev(opts.boffile);
    time.sleep(10)
    print 'Programming complete'
    
    configure_parameters()
    
    #print fpga.listdev()
    #print sorted(fpga.listdev())
    #sys.stdout.flush()
    
    #dump_fft_brams()

    init_10gbe_blocks()
    init_ip_table()
    
    # start recieve code
    raw_input('Press enter to quit: ')
            
    clear_ip_table()
    

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

