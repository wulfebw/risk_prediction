import time
import os
import subprocess

class bcolors:
    """
    source: https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
    """
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_intro(s):
    string = bcolors.BOLD
    string += '#' * 50 + '\n'
    string += 'Running {}...\n'.format(s)
    string += '#' * 50 + '\n'
    string += bcolors.ENDC
    print(string)

def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def build_cmd(key_val_iter, prefix=''):
    cmd = ''
    for (k, v) in key_val_iter:
        if k.startswith(prefix):
            cmd += ' --{} {}'.format(k.replace(prefix,''), v)
    return cmd

class ChangeDir(object):
    """
    source: https://stackoverflow.com/questions/431684/how-do-i-cd-in-python/13197763#13197763
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class Timer(object):    
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        print('Elapsed Time: {}'.format(time.time() - self.start))

def run_cmd(cmd, logfilepath, cmd_dir='.', dry_run=False):
    print('-' * 50) 
    cmd_str = '\n'.join(cmd.split('--'))
    print('{}running cmd: {}\n{}'.format(bcolors.BOLD, bcolors.ENDC, cmd_str))
    print('\nlogging output to: {}'.format(logfilepath))
    if not dry_run:
        with ChangeDir(cmd_dir) as cd, Timer() as t:
            log = open(logfilepath, 'a')
            subprocess.call(cmd, shell=True, stdout=log, stderr=log)
    print('-' * 50) 


