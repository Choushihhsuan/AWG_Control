# **************************************************************************
# Written in May, 2022
# Control program written by KHC in Spring, 2022 @ Lab330 in IAMS, AS
#
# **************************************************************************
# 
# This program handles the communication with the Spectrum Instrument AWG to output the desired RF signal
# to drive the 2D-AOD. The information of the signal waveform is input to the program as an external .txt file.
# In the future, the user may choose to integrate this program into some larger program that also generates the 
# required waveform information. In that case, the user may consider changing the Data input to be some Python datatypes
# directly, instead of writing it to a .txt file and read it out again, which might take some additional time. 
# The current operation mode is cyclic replay. That is, after the Data is finished being played, the program will
# replay the whole waveform (for both channels) over and over again until the user hit esc to terminate the program.
# If the user wants to modify how many times the signal waveform will be replayed, he or she may set llLoops to the desired value
# 
# Please note that prior to executing this program, the user should check that a .txt file describing the properties of the waveform
# is present and ready to be read by this program. 
#
# Documentation for the API as well as a detailed description of the hardware
# can be found in the manual for each device which can be found on our website:
# https://www.spectrum-instrumentation.com/products/details/M4i6622-x8.php
#
# Further information can be found online in the Knowledge Base:
# https://www.spectrum-instrumentation.com/en/knowledge-base-overview
#
# **************************************************************************

# Current problem
# 1. change mode when integrating all the functions
# 2. only 1D currently

from pyspcm import *
from spcm_tools import *
import sys
import msvcrt

class AWG:
    def __init__(self, time, R=625, channelNum=2, refClock=False, refClockFreq=10**7, clockOut=False):
        # open card and check
        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
        # if self.hCard == None:
        #     sys.stdout.write("no card found...\n")
        #     exit(1)

        # reset
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)

        # read type, function and sn and check for D/A card
        self.lCardType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCITYP, byref (self.lCardType))
        self.lSerialNumber = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_PCISERIALNO, byref (self.lSerialNumber))
        self.lFncType = int32 (0)
        spcm_dwGetParam_i32 (self.hCard, SPC_FNCTYPE, byref (self.lFncType))

        self.sCardName = szTypeToName(self.lCardType.value)
        if self.lFncType.value == SPCM_TYPE_AO:
            sys.stdout.write("Found: {0} sn {1:05d}\n".format(self.sCardName, self.lSerialNumber.value))
        else:
            sys.stdout.write("This is an example for D/A cards.\nCard: {0} sn {1:05d} not supported by example\n".
                             format(self.sCardName,self.lSerialNumber.value))
            spcm_vClose(self.hCard);
            exit(1)

        # Configure the reference clock (if required)
        if refClock == True:
            spcm_dwSetParam_i32 (self.hCard, SPC_CLOCKMODE, SPC_CM_EXTREFCLOCK); # Set to reference clock mode
            spcm_dwSetParam_i32 (self.hCard, SPC_REFERENCECLOCK, refClockFreq); # Reference clock that is fed in at the Clock Frequency
            if self.checkExtClock() == True:
                print("Clock has been set\n")
            else:
                print("External Clock not found, please check connection to external clock or set referenceClock to False.\n")
                spcm_dwSetParam_i32 (self.hCard, SPC_CLOCKMODE, SPC_CM_INTPLL) # Enables internal programmable quartz 1
        else:
            print("Using internal clock\n")

        # Set sample rate
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, MEGA(R))
        sys.stdout.write("Sample rate set by the user\n")
        # if ((self.lCardType.value & TYP_SERIESMASK) == TYP_M4IEXPSERIES) or (
        #         (self.lCardType.value & TYP_SERIESMASK) == TYP_M4XEXPSERIES):
        #     spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, MEGA(R))
        #     sys.stdout.write("Sample rate set by the user\n")
        # else:
        #     spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, MEGA(1))
        #     sys.stdout.write("Invalid sample rate. Sample rate 1 \n")

        # Set clock output
        # if clockOut == True:
        #     spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 1)
        # else:
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 0)

        # compute the size of the required memory
        # may need change!!!
        total_sample_L = R * time
        mem_size = 32 * (1 + (total_sample_L // 32))  # the smallest unit of buffer sample points is 32

        # setup the mode
        self.llMemSamples = int64(mem_size)  ## buffer length in number of Data points
        self.llLoops = int64(0)  # number of loops, 0 means continuously (replay mode)
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, SPC_REP_STD_SINGLE)
        spcm_dwSetParam_i64(self.hCard, SPC_CHENABLE, CHANNEL0 | CHANNEL1)  # change here to output on two channels
        spcm_dwSetParam_i64(self.hCard, SPC_MEMSIZE, self.llMemSamples) # buffer length in number of Data points
        spcm_dwSetParam_i64(self.hCard, SPC_LOOPS, self.llLoops) # loop
        spcm_dwSetParam_i64(self.hCard, SPC_ENABLEOUT0, 1)  # output channel 0
        spcm_dwSetParam_i64(self.hCard, SPC_ENABLEOUT1, 1)  # add this line to enable output of channel 1

        # Get the total number of channels
        self.lSetChannels = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_CHCOUNT, byref(self.lSetChannels))
        sys.stdout.write("The number of activated channels is: {:d}\n".format(self.lSetChannels.value))
        self.lBytesPerSample = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_BYTESPERSAMPLE, byref(self.lBytesPerSample))
        sys.stdout.write("The number of bytes per sample is: {:d}\n".format(self.lBytesPerSample.value))

        # setup the trigger mode
        # (SW trigger, no output)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ANDMASK, 0)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_CH_ORMASK0, 0)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_CH_ORMASK1, 0)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIGGEROUT, 0)

        # set the output amplitude (in mV, range: 80~2500)
        self.lChannel = int32(0)
        spcm_dwSetParam_i32(self.hCard, SPC_AMP0, int32(1000))  # 1V for channel 1
        spcm_dwSetParam_i32(self.hCard, SPC_AMP1, int32(1000))  # 1V for channel 2

        # setup software buffer lSetChannels
        self.qwBufferSize = uint64(self.llMemSamples.value * self.lBytesPerSample.value * self.lSetChannels.value)
        sys.stdout.write("The buffer size is: {:d}\n".format(self.qwBufferSize.value))
        # we try to use continuous memory if available and big enough
        self.pvBuffer = c_void_p()
        self.qwContBufLen = uint64(0)
        spcm_dwGetContBuf_i64(self.hCard, SPCM_BUF_DATA, byref(self.pvBuffer), byref(self.qwContBufLen))
        sys.stdout.write("ContBuf length: {0:d}\n".format(self.qwContBufLen.value))
        if self.qwContBufLen.value >= self.qwBufferSize.value:
            sys.stdout.write("Using continuous buffer\n")
        else:
            self.pvBuffer = pvAllocMemPageAligned(self.qwBufferSize.value)
            sys.stdout.write("Using buffer allocated by user program\n")

        sys.stdout.write("llMemSamples: {0:d}\n".format(self.llMemSamples.value))

        self.pnBuffer = cast(self.pvBuffer, ptr16)
        self.lMaxADCValue = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_MAXADCVALUE, byref(self.lMaxADCValue))
        self.dwFS = uint32(self.lMaxADCValue.value)
        self.dwFShalf = uint32(self.dwFS.value // 2)
        sys.stdout.write("lMaxADCValue: {0:d}\n".format(self.lMaxADCValue.value))

    # def checkExtClock(self):
    #     if spcm_dwSetParam_i32(self.hCard, SPC_M2CMD,
    #                            M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER) == ERR_CLOCKNOTLOCKED:
    #         print("External clock not locked. Please check connection\n")
    #         return False
    #     else:
    #         print("External clock locked.\n")
    #         return True

    def transfer_data(self, time, func_x, func_y, R=625):
        end_sample_length = self.llMemSamples.value

        # func -> buffer
        # func = [signal_1, signal_2, ...]
        L = R * time

        for i in range(L): # for each sample
            self.pnBuffer[2*i] = int16 (int(self.dwFShalf.value * func_x[i]))
            # self.pnBuffer[2*i+1] = int16 (int(self.dwFShalf.value * func_y[i]))

        # pad 0 to those unused memories in the buffer (since the buffer length has to be a multiple of 32)
        for i in range(L, end_sample_length, 1):
            self.pnBuffer[i] = int16 (0)


    def execute(self):
        # we define the buffer for transfer and start the DMA transfer
        sys.stdout.write("Starting the DMA transfer and waiting until Data is in board memory\n")
        # set Data transfer
        spcm_dwDefTransfer_i64 (self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), self.pvBuffer, uint64 (0), self.qwBufferSize)
        # start transfer
        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        sys.stdout.write("... Data has been transferred to board memory\n")

        dwError = spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        if dwError != ERR_OK:
            spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
            sys.stdout.write ("... Error: {0:d}\n".format(dwError))
            exit (1)

        # press esc to end this program
        sys.stdout.write ("\n key: ESC ... stop replay and end program\n\n")

        # def lKbhit():
        #     return ord(msvcrt.getch()) if msvcrt.kbhit() else 0
        # while True:
        #     lKey = lKbhit ()
        #     if lKey == 27: # ESC
        #         spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        #         break
        # spcm_vClose (self.hCard);

    def stop_AWG(self):
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        spcm_vClose(self.hCard)
        # exit

