import logging
from ctypes import create_string_buffer, byref, c_void_p, cast, POINTER, c_int32, c_int64, c_uint64, c_int16
from pyspcm import *
from spcm_tools import *

logging.basicConfig(level=logging.INFO)

class AWG:
    def __init__(self, time_s, sample_rate_MHz=625, channels=(0, 1), use_ext_clock=False, ext_clock_freq=10e6):
        self.time_s = time_s
        self.sample_rate = sample_rate_MHz
        self.channels = channels
        self.use_ext_clock = use_ext_clock
        self.ext_clock_freq = ext_clock_freq
        
        self._open_card()
        self._reset_card()
        self._check_card_type()
        self._configure_clock()
        self._configure_samplerate()
        self._configure_memory()
        self._configure_channels()
        self._configure_trigger()
        self._set_amplitudes(1000)
        self._allocate_buffer()

    def _open_card(self):
        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
        if not self.hCard:
            raise RuntimeError("No SPC card found")
        logging.info("Card opened: %s", self.hCard)

    def _reset_card(self):
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)

    def _check_card_type(self):
        card_type = c_int32()
        spcm_dwGetParam_i32(self.hCard, SPC_PCITYP, byref(card_type))
        sn = c_int32()
        spcm_dwGetParam_i32(self.hCard, SPC_PCISERIALNO, byref(sn))
        fnc = c_int32()
        spcm_dwGetParam_i32(self.hCard, SPC_FNCTYPE, byref(fnc))

        if fnc.value != SPCM_TYPE_AO:
            spcm_vClose(self.hCard)
            raise RuntimeError(f"Unsupported card: {szTypeToName(card_type.value)} sn {sn.value}")
        logging.info("Found %s sn %05d", szTypeToName(card_type.value), sn.value)

    def _configure_clock(self):
        if self.use_ext_clock:
            spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE, SPC_CM_EXTREFCLOCK)
            spcm_dwSetParam_i32(self.hCard, SPC_REFERENCECLOCK, int(self.ext_clock_freq))
            logging.info("Using external reference clock at %d Hz", int(self.ext_clock_freq))
        else:
            spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)
            logging.info("Using internal PLL clock")

    def _configure_samplerate(self):
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, MEGA(self.sample_rate))
        logging.info("Sample rate set to %d MHz", self.sample_rate)

    def _configure_memory(self):
        total_samples = int(self.sample_rate * 1e6 * self.time_s)
        self.mem_size = 32 * ((total_samples + 31) // 32)
        self.qwBufferSize = c_uint64(self.mem_size * len(self.channels) * c_int16().value)
        logging.info("Memory: %d samples, buffer size: %d bytes", self.mem_size, self.qwBufferSize.value)

    def _configure_channels(self):
        mask = sum(1 << ch for ch in self.channels)
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, SPC_REP_STD_SINGLE)
        spcm_dwSetParam_i64(self.hCard, SPC_CHENABLE, mask)
        spcm_dwSetParam_i64(self.hCard, SPC_MEMSIZE, self.mem_size)
        spcm_dwSetParam_i64(self.hCard, SPC_LOOPS, 0)
        for ch in self.channels:
            spcm_dwSetParam_i64(self.hCard, SPC_ENABLEOUT0 + ch, 1)
        bytes_per_sample = c_int32()
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_BYTESPERSAMPLE, byref(bytes_per_sample))
        logging.info("Bytes per sample: %d", bytes_per_sample.value)

    def _configure_trigger(self):
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIGGEROUT, 0)

    def _set_amplitudes(self, mV=1000):
        for ch in self.channels:
            spcm_dwSetParam_i32(self.hCard, SPC_AMP0 + ch, int(mV))
        logging.info("Amplitude set to %d mV on channels %s", mV, self.channels)

    def _allocate_buffer(self):
        ptr = c_void_p()
        length = c_uint64()
        spcm_dwGetContBuf_i64(self.hCard, SPCM_BUF_DATA, byref(ptr), byref(length))
        if length.value < self.qwBufferSize.value:
            ptr = pvAllocMemPageAligned(self.qwBufferSize.value)
            logging.info("Allocated program buffer")
        self.pvBuffer = ptr
        self.pnBuffer = cast(self.pvBuffer, POINTER(c_int16))
        max_adc = c_int32()
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_MAXADCVALUE, byref(max_adc))
        self.offset = max_adc.value // 2
        logging.info("ADC max value: %d, offset: %d", max_adc.value, self.offset)

    def transfer_data(self, func_x, func_y):
        n = min(len(func_x), len(func_y), self.mem_size)
        for i in range(n):
            self.pnBuffer[2*i] = int(self.offset * func_x[i])
            self.pnBuffer[2*i + 1] = int(self.offset * func_y[i])
        for i in range(2 * n, self.mem_size * len(self.channels)):
            self.pnBuffer[i] = 0
        logging.info("Data transferred into buffer: %d samples", n)

    def execute(self):
        spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, self.pvBuffer, 0, self.qwBufferSize)
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        err = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        if err != ERR_OK:
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
            raise RuntimeError(f"Error starting output: {err}")
        logging.info("Output started")

    def stop(self):
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        spcm_vClose(self.hCard)
        logging.info("Output stopped and card closed")
