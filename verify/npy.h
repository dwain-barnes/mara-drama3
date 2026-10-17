#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// numpy .npy with a fixed little-endian dtype in C order
template <typename T>
static std::vector<T> load_npy(const std::string & path, const char * dtype, std::vector<int64_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    char magic[8];
    f.read(magic, 8);
    if (!f || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not an npy file\n", path.c_str());
        exit(1);
    }
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h;
        f.read((char *) &h, 2);
        hlen = h;
    } else {
        f.read((char *) &hlen, 4);
    }
    std::string hdr(hlen, ' ');
    f.read(hdr.data(), hlen);
    if (hdr.find(dtype) == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos) {
        fprintf(stderr, "%s: expected %s in C order, header %s\n", path.c_str(), dtype, hdr.c_str());
        exit(1);
    }

    shape.clear();
    size_t n = 1;
    const char * p = hdr.c_str() + hdr.find("'shape': (") + 10;
    while (*p != ')') {
        char * end;
        long long d = strtoll(p, &end, 10);
        shape.push_back(d);
        n *= (size_t) d;
        p = end;
        while (*p == ',' || *p == ' ') {
            p++;
        }
    }

    std::vector<T> data(n);
    f.read((char *) data.data(), n * sizeof(T));
    if (!f) {
        fprintf(stderr, "%s: short read\n", path.c_str());
        exit(1);
    }
    return data;
}
