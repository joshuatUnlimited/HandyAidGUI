import struct

def read_gguf_string_bytes(data, offset=0):
    if offset + 8 > len(data):
        raise ValueError("Truncated GGUF string length")
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    end = offset + length
    if end > len(data):
        raise ValueError("Truncated GGUF string data")
    return data[offset:end].decode("utf-8", errors="replace"), end

def read_gguf_metadata(path, wanted_keys):
    wanted = set(wanted_keys)
    fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        header = f.read(24)
        if len(header) != 24 or header[:4] != b"GGUF":
            raise ValueError("Not a GGUF file")
        version, tensor_count, kv_count = struct.unpack_from("<IQQ", header, 4)
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        def skip_value(vtype):
            if vtype in fixed_sizes:
                f.seek(fixed_sizes[vtype], 1); return
            if vtype == 8:
                raw=f.read(8)
                if len(raw)!=8: raise ValueError("Truncated GGUF string")
                f.seek(struct.unpack("<Q",raw)[0],1); return
            if vtype == 9:
                item_type=struct.unpack("<I",f.read(4))[0]
                n=struct.unpack("<Q",f.read(8))[0]
                for _ in range(n): skip_value(item_type)
                return
            raise ValueError(f"Unsupported GGUF metadata type {vtype}")
        result={}
        for _ in range(kv_count):
            raw=f.read(8)
            if len(raw)!=8: raise ValueError("Truncated GGUF metadata key length")
            key_len=struct.unpack("<Q",raw)[0]
            key=f.read(key_len).decode("utf-8",errors="replace")
            raw=f.read(4)
            if len(raw)!=4: raise ValueError("Truncated GGUF metadata value type")
            vtype=struct.unpack("<I",raw)[0]
            if key in wanted and vtype == 8:
                raw=f.read(8)
                if len(raw)!=8: raise ValueError("Truncated GGUF string")
                n=struct.unpack("<Q",raw)[0]
                result[key]=f.read(n).decode("utf-8",errors="replace")
            else:
                skip_value(vtype)
        return result

class GGUFModelMixin:
    @staticmethod
    def _read_gguf_string_bytes(data, offset=0):
        return read_gguf_string_bytes(data, offset)
    @classmethod
    def _read_gguf_metadata(cls, path, wanted_keys):
        return read_gguf_metadata(path, wanted_keys)
