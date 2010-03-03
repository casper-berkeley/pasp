#!/usr/bin/env python2.6

import corr, time, struct, sys, logging, socket

numtengbe=4
numips=4
numchannels=16
packetsize=1024

reorder_order=3

fft_coeffs=[4096,4096,4096,4096,4096,4096,4096,4096]

# calculate the sync period
# LCM(reorder orders)*(FFTSize/simultaneousinputs)*numtengbe*packetsize
sync_period=reorder_order*numchannels*numtengbe*packetsize

gbe_base='pasp_dist_gbe_ten_GbE'
ip_reg_base='pasp_dist_gbe_ip_ctr_reg_ip'
port_reg_base='pasp_dist_gbe_ip_ctr_reg_port'

transmit_ip=10*(2**24) + 60 #10.0.0.60
transmit_port=60000

mac_base=(2<<40) + (2<<32)
fabric_ip=10*(2**24) + 30 #10.0.0.30
fabric_port=60000 


katcp_port=7147
boffile='pasp_4i16c1ks.bof'

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    exit()

def exit_clean():
    print 'NO FAILURES'
    #try:
    #    for f in fpgas: f.stop()
    #except: pass
    #exit()
    
def dump_adc_brams():
    #trigger the adcscope
    fpga.write_int('pasp_reg_adcscope_trigger',1)
    time.sleep(1)
    fpga.write_int('pasp_reg_adcscope_trigger',0)
    
    #read the bram data
    adcscope1 = struct.unpack('>2048b',fpga.read('pasp_bram_adcscope_1',2048))
    adcscope2 = struct.unpack('>2048b',fpga.read('pasp_bram_adcscope_2',2048))
    #print adcscope1
    #print adcscope2

def dump_fft_brams():
    fftscope1 = struct.unpack('>2048b',fpga.read('pasp_scope_output1_bram',2048))
    fftscope2 = struct.unpack('>2048b',fpga.read('pasp_scope_output2_bram',2048))
    fftscope3 = struct.unpack('>2048b',fpga.read('pasp_scope_output3_bram',2048))
    fftscope4 = struct.unpack('>2048b',fpga.read('pasp_scope_output3_bram',2048))
    
if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('pasp_init.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
 
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'No ROACH board specified. Defaulting to ROACH01'
        roach = 'ROACH01'
        #print 'Please specify a ROACH board. \nExiting.'
        #exit()
    else:
        roach = args[0]

try:
    loggers = []
    lh=corr.log_handlers.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s on port %i... '%(roach,katcp_port)),
    fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout=10,logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s on port %i.\n'%(roach,katcp_port)
        exit_fail()
        
        
    print 'Available pasp boffiles:'  
    for testboffile in fpga.listbof():
        if 'pasp' in testboffile:
            print testboffile
          
    print 'Programming the fpga with %s'%(boffile)          
    fpga.progdev(boffile);
    time.sleep(10)
    print 'Programming complete'
    
    #print fpga.listdev()
    #print sorted(fpga.listdev())
    #sys.stdout.flush()
    
    # initialize the fpga sync
    print 'Configuring a sync period of %d'%(sync_period)
    fpga.write_int('pasp_reg_sync_period',sync_period)
    
    dump_adc_brams()
    
    # initialize the fft shift schedule
    fpga.write_int('pasp_reg_fft_shift',0xffffffff)
    
    # initialize the scaling parameters
    struct.pack('>8L',*fft_coeffs)
    fpga.write('pasp_scale_ctr_pol0_even_bram',fft_coeffs_string)
    fpga.write('pasp_scale_ctr_pol0_odd_bram',fft_coeffs_string)

    # initialize the scaling bit selection
    fpga.write_int('pasp_reg_output_bitselect_pol1',1)
    fpga.write_int('pasp_reg_output_bitselect_pol2',1)
    
    dump_fft_brams()

    
    # initialize the 10gbe ports
    for i in range(0,numtengbe):
        print 'initializing '+gbe_base+str(i)
        fpga.tap_start(gbe_base+str(i),mac_base+fabric_ip,fabric_ip,fabric_port)
        
            
    # initialize the ip table
    print 'initializing ip table'
    for i in range(0,numips):
        fpga.write_int(ip_reg_base+str(i),transmit_ip+i)
        fpga.write_int(port_reg_base+str(i),transmit_port)
    
    # start recieve code
        
    # zero the ip table
    print 'zeroing ip table'
    for i in range(0,numips):
        fpga.write_int(ip_reg_base+str(i),0)
    
    

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

