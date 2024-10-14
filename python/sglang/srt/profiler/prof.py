from ctypes import *
import os
import time

class SGlangProf:
    def __init__(self):
        self.use_nvtx = os.getenv('SGLANG_PROF_NVTX') is not None
        if self.use_nvtx:
            self.lib = cdll.LoadLibrary("libnvToolsExt.so")
            self.lib.nvtxRangePushA.argtypes = [c_char_p]
            self.lib.nvtxRangePushA.restype = c_int
            self.lib.nvtxRangePop.restype = c_int

        self.use_roctx = os.getenv('SGLANG_PROF_ROCTX') is not None
        if self.use_roctx:
            self.lib = cdll.LoadLibrary("libroctracer64.so")
            self.lib.roctxRangePushA.argtypes = [c_char_p]
            self.lib.roctxRangePushA.restype = c_int
            self.lib.roctxRangePop.restype = c_int
        self.tm = time.perf_counter()
        self.tm_row = {}
        self.tm_clms = []
        self.auto_push_flag = False

    def ResetTimer(self, file):
        if any(self.tm_row):
            with open(file, 'a') as f:
                for clm in self.tm_clms:
                    f.write(str(self.tm_row[clm]) + ',')
                    self.tm_row[clm] = -1
                f.write('\n')
        self.tm = time.perf_counter()
        
    def AddTimer(self, clm):
        new_tm = time.perf_counter()
        self.tm_row[clm] = (new_tm - self.tm)
        if clm not in self.tm_clms:
            self.tm_clms.append(clm)
            print('profile new clm:', clm)
        self.tm = time.perf_counter()


    def ProfRangePush(self, message):
        if profile.use_nvtx:
            profile.lib.nvtxRangePushA(message.encode('utf-8'))
        if profile.use_roctx:
            profile.lib.roctxRangePushA(message.encode('utf-8'))

    def ProfRangePop(self):
        if profile.use_nvtx:
            profile.lib.nvtxRangePop()
        if profile.use_roctx:
            profile.lib.roctxRangePop()

    def ProfRangeAutoPush(self, message):
        self.ProfRangePop()
        self.ProfRangePush(message)


profile = SGlangProf()


if __name__ == '__main__':
    profile.ResetTimer('test.csv')
    time.sleep(1)
    profile.AddTimer('test1')
    time.sleep(1)
    profile.AddTimer('test2')
    profile.ResetTimer('test.csv')