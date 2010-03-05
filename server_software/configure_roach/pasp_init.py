#!/usr/bin/env python2.6

import corr, time, struct, sys, logging, socket, re


# fft scaling configuration
fft_coeffs=[4096,4096,4096,4096,4096,4096,4096,4096]

# ip table configuration
ip_table=['0.0.0.0','0.0.0.0','0.0.0.0','0.0.0.0']
port_table=[60000,60000,60000,60000]




ip_reg_base='pasp_dist_gbe_ip_ctr_reg_ip'
port_reg_base='pasp_dist_gbe_ip_ctr_reg_port'


gbe_base='pasp_dist_gbe_ten_GbE'

mac_base=(2<<40) + (2<<32)
fabric_ip=struct.unpack('=L',socket.inet_aton('10.0.0.30'))[0] #convert ip to long
fabric_port=60000 


katcp_port=7147


def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    exit()

def exit_clean():
    #print 'NO FAILURES'
    try:
        for f in fpgas: f.stop()
    except: pass
    exit()
    
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
    fftscope1 = struct.unpack('>2048b',fpga.read('pasp_scope_output1_bram',2048))
    fftscope2 = struct.unpack('>2048b',fpga.read('pasp_scope_output2_bram',2048))
    fftscope3 = struct.unpack('>2048b',fpga.read('pasp_scope_output3_bram',2048))
    fftscope4 = struct.unpack('>2048b',fpga.read('pasp_scope_output3_bram',2048))
    
    print fftscope1
    print fftscope2
    print fftscope3
    print fftscope4
    
if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('pasp_init.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
 
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'No ROACH board specified. Defaulting to ROACH01'
        roach = 'ROACH01'
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
        
        
    #print 'Available pasp boffiles:'  
    #for testboffile in fpga.listbof():
    #    if 'pasp' in testboffile:
    #        print testboffile
    
    boffile='pasp_4i16c1024s4g.bof'
    
    #extract build configuration info from boffile name
    m=re.search('pasp_([\d]+)i([\d]+)c([\d]+)s([\d]+)g',boffile)
    
    # pasp build configuration
    numtengbe=4
    numips=int(m.group(1))
    numchannels=int(m.group(2))
    packetsize=int(m.group(3))*1024
    reorder_order=3
    
    print('Got numips %d numchannels %d packetsize %d'%(numips,numchannels,packetsize))

    # calculate the sync period
    # LCM(reorder orders)*(FFTSize/simultaneousinputs)*numtengbe*packetsize
    sync_period=reorder_order*numchannels*numtengbe*packetsize
            
    logger.debug('Clearing the fpga')       
    fpga.progdev('')
    time.sleep(1)
    logger.debug('Programming the fpga with %s'%(boffile))   
    print 'Programming the fpga with %s'%(boffile)  
    fpga.progdev(boffile);
    time.sleep(10)
    print 'Programming complete'
    
    #print fpga.listdev()
    print sorted(fpga.listdev())
    #sys.stdout.flush()
    
    # initialize the fpga sync
    logger.debug('Configuring a sync period of %d'%(sync_period))
    fpga.write_int('pasp_reg_sync_period',sync_period)
    
    #dump_adc_brams()
    
    # initialize the fft shift schedule
    fpga.write_int('pasp_reg_fft_shift',0xffffffff)
    
    # initialize the scaling parameters
    fft_coeffs_string = struct.pack('>8L',*fft_coeffs)
    fpga.write('pasp_scale_ctr_pol0_even_bram',fft_coeffs_string)
    fpga.write('pasp_scale_ctr_pol0_odd_bram',fft_coeffs_string)

    # initialize the scaling bit selection
    fpga.write_int('pasp_reg_output_bitselect_pol1',1)
    fpga.write_int('pasp_reg_output_bitselect_pol2',1)
    
    #dump_fft_brams()

    
    # initialize the 10gbe ports
    for i in range(0,numtengbe):
        logger.debug('Initializing '+gbe_base+str(i))
        fpga.tap_start(gbe_base+str(i),mac_base+fabric_ip,fabric_ip,fabric_port)
        
            
    # initialize the ip table
    logger.debug('Initializing ip table')
    for i in range(0,numips):
        fpga.write_int(ip_reg_base+str(i),struct.unpack('=L',socket.inet_aton(ip_table[i]))[0])
        fpga.write_int(port_reg_base+str(i),port_table[i])
    
    # start recieve code
        
    # zero the ip table
    print 'Zeroing ip table'
    for i in range(0,numips):
        fpga.write_int(ip_reg_base+str(i),0)
            
    
    

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

