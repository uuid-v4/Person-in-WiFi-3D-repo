"""Parse the ESP32 recorder's live USB stream: 0xC5 0x1A 'R' + 118-byte record + 1-byte sum."""
import numpy as np

MAGIC = b'\xc5\x1aR'
REC_DTYPE = np.dtype([('t_us', '<u4'), ('seq', '<u4'), ('rssi', 'i1'), ('rate', 'u1'), ('agc', 'u1'),
                      ('fft', 'i1'), ('sig_mode', 'u1'), ('flags', 'u1'), ('csi', 'i1', (104,))])
REC = REC_DTYPE.itemsize
FRAME = len(MAGIC) + REC + 1


class FrameParser:
    def __init__(self):
        self.buf = bytearray()
        self.bad = 0

    def feed(self, data):
        """Append raw serial bytes; return list of record bytes (each REC long) with a valid checksum."""
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                del self.buf[:max(0, len(self.buf) - 2)]
                return out
            if len(self.buf) - i < FRAME:
                del self.buf[:i]
                return out
            rec = bytes(self.buf[i + 3:i + 3 + REC])
            if sum(rec) & 0xFF == self.buf[i + 3 + REC]:
                out.append(rec)
                del self.buf[:i + FRAME]
            else:
                self.bad += 1
                del self.buf[:i + 1]
