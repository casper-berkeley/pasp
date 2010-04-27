#!/usr/bin/env python

import struct, socket, sys, corr, logging, time, re
import pasp
import pasp_plot


# default boffile
defaultbof='pasp_4i16c1024s4g.bof'

# define katcp port to connect
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

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('pasp_init.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.set_defaults(interactive_mode=False)
    p.set_defaults(plot_mode=False)
    p.set_defaults(boffile=defaultbof)
    p.add_option('-i','--interactive', help='select boffile interactively', action='store_true', dest='interactive_mode')
    p.add_option('-f','--file', help='use specified boffile', action='store', type='string', dest='boffile')
    p.add_option('-p','--plot', help='don''t configure roach, just connect and plot fft brams', action='store_true', dest='plot_mode')
 
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'No ROACH board specified. Defaulting to ROACH01'
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
            if 'pasp' in testboffile:
                print testboffile
        opts.boffile=raw_input('Select a boffile: ')

    new_pasp = pasp.pasp(fpga,opts.boffile,logger)
            
    if opts.plot_mode == True:
        pasp_plot.plot_fft_brams(new_pasp)
    
    else:
        new_pasp.reprogram_fpga()


except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()