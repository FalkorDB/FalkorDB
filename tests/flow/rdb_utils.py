# CRC-64 (Jones variant) - the checksum Redis stamps at the end of an RDB file
# and into DUMP/RESTORE payload footers. reflected CRC, so the right-shift form
# uses the reflected Jones polynomial (reflect(0xad93d23594c935a9))
def crc64(data):
    POLY = 0x95ac9329ac4bc9b5
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ POLY if (crc & 1) else (crc >> 1)
    return crc & 0xFFFFFFFFFFFFFFFF
