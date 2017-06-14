import time
import os
import subprocess

def print_intro(s):
    print('#' * 50)
    print('Running {}...'.format(s))
    print('#' * 50)

def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def build_cmd(key_val_iter, prefix=''):
    cmd = ''
    for (k, v) in key_val_iter:
        if k.startswith(prefix):
            cmd += ' --{} {}'.format(k.replace(prefix,''), v)
    return cmd

def run_cmd(cmd, logfilepath):
    print('-' * 50) 
    cmd_str = '\n'.join(cmd.split('--'))
    print('running cmd:\n{}'.format(cmd_str))
    print('\nlogging output to: {}'.format(logfilepath))
    print('-' * 50) 
    log = open(logfilepath, 'a')
    subprocess.call(cmd, shell=True, stdout=log, stderr=log)

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print('Elapsed Time: {}'.format(time.time() - self.start))