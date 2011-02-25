import struct, socket, sys, corr, logging, time, re


defaultbof='gbetest_noxblocks_2011_Jan_20_1830.bof' # default boffile
katcp_port=7147                                     # define katcp port to connect


mac_base=(2<<40) + (2<<32)
fabric_ip=struct.unpack('!L',socket.inet_aton('10.0.0.31'))[0] #convert ip to long
fabric_port=33107

#dest_ip=struct.unpack('!L',socket.inet_aton('10.0.0.30'))[0]
dest_ip=struct.unpack('!L',socket.inet_aton('10.0.0.4'))[0]
dest_port=33107

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
        print 'No ROACH board specified. Defaulting to roach01'
        roach = 'roach01'
    else:
        roach = args[0]

try:
    lh=corr.log_handlers.DebugLogHandler()
    global logger 
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
            if 'gbetest' in testboffile:
                print testboffile
        opts.boffile=raw_input('Select a boffile: ')

    # program the fpga with selected boffile
    logger.debug('Programming the fpga with %s'%(opts.boffile))   
    print 'Programming the fpga with %s'%(opts.boffile)  
    fpga.progdev(opts.boffile);
    time.sleep(10)
    print 'Programming complete'
    
    #print fpga.listdev();
    
    fpga.write_int('pkt_size',1024);
    #fpga.write_int('pkt_wait',1300);
    fpga.write_int('pkt_wait',450);
    fpga.write_int('dest_ip',dest_ip);
    fpga.write_int('dest_port',dest_port);
    
    fpga.write_int('pkt_rst',1)
    fpga.tap_start('ten_GbE','ten_GbE',mac_base+fabric_ip,fabric_ip,fabric_port)
    fpga.tap_start('ten_GbE1','ten_GbE1',mac_base+fabric_ip+1,fabric_ip+1,fabric_port)
    fpga.tap_start('ten_GbE2','ten_GbE2',mac_base+fabric_ip+2,fabric_ip+2,fabric_port)
    fpga.tap_start('ten_GbE3','ten_GbE3',mac_base+fabric_ip+3,fabric_ip+3,fabric_port)
    time.sleep(2)
    fpga.write_int('pkt_rst',0)
    
    time.sleep(2)
    print fpga.read_int('pkt_ctr')
    time.sleep(2)
    print fpga.read_int('pkt_ctr')
    time.sleep(100)
    print fpga.read_int('pkt_ctr')
    fpga.write_int('dest_port',0);
    fpga.write_int('dest_ip',0);
    time.sleep(2)
    print fpga.read_int('pkt_ctr')
    

    fpga.progdev('');

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()
