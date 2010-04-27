import corr, time, struct, sys, logging, socket, re, itertools
import math


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
ip_table=['10.0.0.2','10.0.0.2','10.0.0.2','10.0.0.2']
port_table=[33107,33107,33107,33107]


ip_reg_base='pasp_dist_gbe_ip_ctr_reg_ip'
port_reg_base='pasp_dist_gbe_ip_ctr_reg_port'


gbe_base='pasp_dist_gbe_ten_GbE'

mac_base=(2<<40) + (2<<32)
fabric_ip=struct.unpack('!L',socket.inet_aton('10.0.0.30'))[0] #convert ip to long
fabric_port=33107

class pasp:
    
    def __init__(self,fpga,boffile,logger):
        self.fpga=fpga
        self.boffile=boffile
        self.logger=logger
        
        # extract build configuration info from boffile name
        m=re.search('pasp_([\d]+)i([\d]+)c([\d]+)s([\d]+)g',boffile)
        
        # get pasp build configuration from boffile name
        self.numips=int(m.group(1))
        self.numchannels=int(m.group(2))
        self.packetsize=int(m.group(3))
        self.numtengbe=int(m.group(4))
        self.reorder_order=3
        
        logger.debug('Got numips %d numchannels %d packetsize %d numtengbe %d'%(self.numips,self.numchannels,self.packetsize,self.numtengbe))
        
        # calculate the sync period
        # LCM(reorder orders)*(FFTSize/simultaneousinputs)*numtengbe*packetsize
        self.sync_period=self.reorder_order*self.numchannels*self.numtengbe*self.packetsize
        
    
    # reprogram the fpga with the boffile
    def reprogram_fpga(self):
        # program the fpga with selected boffile
        self.logger.debug('Programming the fpga with %s'%(self.boffile))   
        print 'Programming the fpga with %s'%(self.boffile)  
        self.fpga.progdev(self.boffile);
        time.sleep(10)
        print 'Programming complete'
        
        self.configure_parameters()
        
        #print fpga.listdev()
        #print sorted(fpga.listdev())
        #sys.stdout.flush()
        
        #dump_fft_brams()

        self.init_10gbe_blocks()
        self.init_ip_table()
        
        # start recieve code
        raw_input('Press enter to quit: ')
                
        self.clear_ip_table()


    def interleave(self, *args):
        retlist=[]
        for idx in range(0, max(len(arg) for arg in args)):
            for arg in args:
                try:
                    retlist.append(arg[idx])
                except IndexError:
                    continue
        return retlist

    
    # functions for dumping data    
    def dump_adc_brams(self):
        #trigger the adcscope
        fpga.write_int('pasp_reg_adcscope_trigger',1)
        time.sleep(1)
        fpga.write_int('pasp_reg_adcscope_trigger',0)
        
        #read the bram data
        adcscope1 = struct.unpack('>2048b',fpga.read('pasp_bram_adcscope_1',2048))
        adcscope2 = struct.unpack('>2048b',fpga.read('pasp_bram_adcscope_2',2048))
        print adcscope1
        print adcscope2
        
    def get_fft_brams_power(self):
        retlist=[]
        fftscope_re, fftscope_im = self.get_fft_brams()
        for i in range(0,self.numchannels):
            retlist.append(math.pow(fftscope_re[i],2) + math.pow(fftscope_im[i],2))
        return retlist
        
    def get_fft_brams(self):
        fftscope_even_re = struct.unpack('>8l',self.fpga.read('pasp_scope_output1_bram',8*4))
        fftscope_even_im = struct.unpack('>8l',self.fpga.read('pasp_scope_output2_bram',8*4))
        fftscope_odd_re = struct.unpack('>8l',self.fpga.read('pasp_scope_output3_bram',8*4))
        fftscope_odd_im = struct.unpack('>8l',self.fpga.read('pasp_scope_output4_bram',8*4))
        fftscope_re = self.interleave(fftscope_even_re,fftscope_odd_re)
        fftscope_im = self.interleave(fftscope_even_im,fftscope_odd_im)

        return fftscope_re, fftscope_im

    def dump_fft_brams(self):
        fftscope_re, fftscope_im = self.get_fft_brams()
        print fftscope_re
        print fftscope_im

    
    # configure the roach parameters
    def configure_parameters(self):
        # initialize the fpga sync
        self.logger.debug('Configuring a sync period of %d'%(self.sync_period))
        self.fpga.write_int('pasp_reg_sync_period',self.sync_period)
        
        # initialize the fft shift schedule
        self.fpga.write_int('pasp_reg_fft_shift',fft_shift)
        #fpga.write_int('pasp_reg_fft_shift',0x0)
        
        # initialize the scaling parameters
        fft_coeffs_string = struct.pack('>8L',*fft_coeffs)
        self.fpga.write('pasp_scale_ctr_pol0_even_bram',fft_coeffs_string)
        self.fpga.write('pasp_scale_ctr_pol0_odd_bram',fft_coeffs_string)
        self.fpga.write('pasp_scale_ctr_pol1_even_bram',fft_coeffs_string)
        self.fpga.write('pasp_scale_ctr_pol1_odd_bram',fft_coeffs_string)

        # initialize the scaling bit selection
        self.fpga.write_int('pasp_reg_output_bitselect_pol1',bitselect_pol1)
        self.fpga.write_int('pasp_reg_output_bitselect_pol2',bitselect_pol2)

    # initialize 10gbe
    def init_10gbe_blocks(self):
        # initialize the 10gbe ports
        for i in range(0,self.numtengbe):
            self.logger.debug('Initializing '+gbe_base+str(i))
            self.fpga.tap_start('gbe'+str(i),gbe_base+str(i),mac_base+fabric_ip+i,fabric_ip+i,fabric_port)
        
    # set ip addresses
    def init_ip_table(self):
        # reset the packet counter
        self.fpga.write_int('pasp_dist_gbe_rst_packet_count',1)
        time.sleep(1)
        self.fpga.write_int('pasp_dist_gbe_rst_packet_count',0)
        # initialize the ip table
        self.logger.debug('Initializing ip table')
        # initialize port first so packets don't get sent to port 0
        for i in range(0,self.numips):
            self.fpga.write_int(port_reg_base+str(i),port_table[i])
            self.fpga.write_int(ip_reg_base+str(i),struct.unpack('!L',socket.inet_aton(ip_table[i]))[0])

    # clear ip addresses to stop packet flow
    def clear_ip_table(self):
        # zero the ip table
        print('Zeroing ip table')
        for i in range(0,self.numips):
            self.fpga.write_int(ip_reg_base+str(i),0)
        # read the packet count
        for i in range(0,self.numtengbe):
            print('Sent %d packets on gbe%d'%(self.fpga.read_int('pasp_dist_gbe_reg_packet_count%d'%(i)),i))
        
        

        


