import multiprocessing

def get_thread_count(thread_count = 'auto'):
    if thread_count and thread_count != 'auto':
        return int(thread_count)
    return multiprocessing.cpu_count()