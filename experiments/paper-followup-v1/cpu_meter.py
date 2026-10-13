"""Case-process-group CPU accounting. No third-party packages.

Darwin: proc_pid_rusage v1 self CPU + reaped-child CPU, across all current
group members (including zombies). Linux backend exists for development tests.
Reject snapshots spanning a membership/child-accounting transition.
"""
import ctypes
import errno
import os
from pathlib import Path
import sys
import time


class Usage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user', 'system', 'pkg_wakeups', 'interrupt_wakeups', 'pageins',
        'wired', 'resident', 'footprint', 'start', 'exit',
        'child_user', 'child_system', 'child_pkg_wakeups',
        'child_interrupt_wakeups', 'child_pageins', 'child_elapsed')]


class Meter:
    def __init__(self, pgid):
        self.pgid = pgid
        self.last = 0.0
        self.retries = 0
        if sys.platform == 'darwin':
            self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            self.lib.proc_listpgrppids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            self.lib.proc_listpgrppids.restype = ctypes.c_int
            self.lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            self.lib.proc_listchildpids.restype = ctypes.c_int
            self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self.lib.proc_pid_rusage.restype = ctypes.c_int
        elif sys.platform.startswith('linux'):
            self.ticks = os.sysconf('SC_CLK_TCK')
            self.proc_paths = {pgid: Path('/proc') / os.readlink('/proc/self')}
            self.host_group = int((Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split())[2])
        else:
            raise RuntimeError('Supported platforms: macOS; Linux for development tests')

    def members(self):
        if sys.platform == 'darwin':
            capacity = 256
            while capacity <= 65536:
                buffer = (ctypes.c_int * capacity)()
                ctypes.set_errno(0)
                count = self.lib.proc_listpgrppids(self.pgid, buffer, ctypes.sizeof(buffer))
                error = ctypes.get_errno()
                if count < 0 or (count == 0 and error):
                    raise OSError(error, 'Cannot enumerate owned process group')
                if count < capacity:
                    return sorted(pid for pid in buffer[:count] if pid > 0)
                capacity *= 2
            raise RuntimeError('Process group enumeration overflow')
        found = []
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == self.host_group:
                    pid = self.local_pid(path.parent)
                    self.proc_paths[pid] = path.parent
                    found.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                pass
        return sorted(found)

    def local_pid(self, path):
        # /proc may be mounted from an outer PID namespace in development.
        for line in (path / 'status').read_text().splitlines():
            if line.startswith('NSpid:'):
                return int(line.split()[-1])
        return int(path.name)

    def usage(self, pid):
        if sys.platform == 'darwin':
            u = Usage()
            if self.lib.proc_pid_rusage(pid, 1, ctypes.byref(u)) != 0:
                error = ctypes.get_errno()
                raise OSError(error, 'proc_pid_rusage failed for PID %d' % pid)
            return dict(own=(u.user + u.system) / 1e9,
                child=(u.child_user + u.child_system) / 1e9,
                identity=u.start, exited=bool(u.exit), rss=u.resident)
        fields = (self.proc_paths.get(pid, Path('/proc') / str(pid)) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(own=(int(fields[11]) + int(fields[12])) / self.ticks,
            child=(int(fields[13]) + int(fields[14])) / self.ticks,
            identity=int(fields[19]), exited=fields[0] == 'Z',
            rss=max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE'))

    def descendants(self, parent):
        if sys.platform == 'darwin':
            buffer = (ctypes.c_int * 1024)()
            ctypes.set_errno(0)
            count = self.lib.proc_listchildpids(parent, buffer, ctypes.sizeof(buffer))
            error = ctypes.get_errno()
            if count < 0 or (count == 0 and error) or count >= 1024:
                raise OSError(error, 'Cannot enumerate case descendants')
            children = [pid for pid in buffer[:count] if pid > 0]
        else:
            children = []
            for path in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields = path.read_text().rsplit(')', 1)[1].split()
                    if int(fields[1]) == int(self.proc_paths.get(parent, Path('/proc') / str(parent)).name):
                        pid = self.local_pid(path.parent)
                        self.proc_paths[pid] = path.parent
                        children.append(pid)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        result = set(children)
        for child in children:
            result.update(self.descendants(child))
        return result

    def snapshot(self):
        for _ in range(40):
            try:
                ids = self.members()
                if self.pgid not in ids:
                    raise RuntimeError('Accounting supervisor left its process group')
                if set(ids) != {self.pgid} | self.descendants(self.pgid):
                    self.retries += 1
                    time.sleep(0.001)
                    continue
                first = {pid: self.usage(pid) for pid in ids}
                second = {pid: self.usage(pid) for pid in ids}
                if ids != self.members() or any(
                    first[pid]['identity'] != second[pid]['identity'] or
                    first[pid]['child'] != second[pid]['child'] for pid in ids):
                    self.retries += 1
                    continue
                # Supervisor overhead is excluded. Its waited-child CPU is kept.
                total = sum(v['own'] + v['child'] for v in second.values()) - second[self.pgid]['own']
                if total + 0.05 < self.last:
                    raise RuntimeError('CPU accounting decreased: %.6f -> %.6f; refusing an unreliable budget' % (self.last, total))
                self.last = max(self.last, total)
                return dict(cpu_seconds=self.last,
                    resident_bytes=sum(v['rss'] for pid, v in second.items() if pid != self.pgid),
                    process_count=len(ids) - 1, snapshot_retries=self.retries)
            except (FileNotFoundError, ProcessLookupError):
                self.retries += 1
            except OSError as exc:
                if exc.errno not in (errno.ESRCH, errno.ENOENT):
                    raise
                self.retries += 1
            time.sleep(0.001)
        raise RuntimeError('Could not obtain a stable CPU-accounting snapshot')
