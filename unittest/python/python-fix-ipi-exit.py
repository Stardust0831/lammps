#!/usr/bin/env python3
"""Exercise fix ipi shutdown with a small socket peer and one LAMMPS rank.

Run directly with: python3 python-fix-ipi-exit.py --lmp /path/to/lmp -v
No i-PI installation, external potential, or LAMMPS Python module is needed.
"""
import argparse
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest


def header(name):
    return name.encode('ascii').ljust(12, b' ')


def receive(sock, count):
    result = bytearray()
    while len(result) < count:
        block = sock.recv(count - len(result))
        if not block:
            raise EOFError('LAMMPS closed the socket before the reply was complete')
        result.extend(block)
    return bytes(result)


class FixIPIExit(unittest.TestCase):
    executable = None

    def exchange(self, stage, message='EXIT'):
        with tempfile.TemporaryDirectory(prefix='lammps-ipi-exit-') as directory:
            work = Path(directory)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind(('127.0.0.1', 0))
                server.listen(1)
                server.settimeout(15)
                port = server.getsockname()[1]
                (work / 'input.lmp').write_text('''units metal
atom_style atomic
atom_modify map yes
boundary p p p
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single -0.5 0 0
create_atoms 1 single 0.5 0 0
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff * * 0.1 1.0
neighbor 0.3 bin
thermo 1
fix driver all ipi 127.0.0.1 %d
run 10
''' % port)
                environment = os.environ.copy()
                environment['OMP_NUM_THREADS'] = '1'
                with (work / 'output.log').open('w') as output:
                    process = subprocess.Popen(
                        [self.executable, '-log', 'none', '-in', 'input.lmp'],
                        cwd=str(work), stdout=output, stderr=subprocess.STDOUT,
                        env=environment)
                    try:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(15)
                            connection.sendall(header('STATUS'))
                            self.assertEqual(receive(connection, 12), header('READY'))
                            if stage != 'ready':
                                # i-PI sends atomic units; use the conversion in fix ipi.
                                length = 10.0 / 0.52917721
                                cell = [length, 0, 0, 0, length, 0, 0, 0, length]
                                inverse = [1.0 / length, 0, 0, 0, 1.0 / length, 0,
                                           0, 0, 1.0 / length]
                                positions = [-0.5 / 0.52917721, 0, 0,
                                             0.5 / 0.52917721, 0, 0]
                                connection.sendall(header('POSDATA')
                                                   + struct.pack('=18d', *(cell + inverse))
                                                   + struct.pack('=i', 2)
                                                   + struct.pack('=6d', *positions))
                                connection.sendall(header('STATUS'))
                                self.assertEqual(receive(connection, 12), header('HAVEDATA'))
                            if stage == 'after_force':
                                connection.sendall(header('GETFORCE'))
                                self.assertEqual(receive(connection, 12), header('FORCEREADY'))
                                energy = struct.unpack('=d', receive(connection, 8))[0]
                                count = struct.unpack('=i', receive(connection, 4))[0]
                                self.assertEqual(count, 2)
                                forces = struct.unpack('=6d', receive(connection, 48))
                                virial = struct.unpack('=9d', receive(connection, 72))
                                extra = struct.unpack('=i', receive(connection, 4))[0]
                                self.assertGreaterEqual(extra, 0)
                                receive(connection, extra)
                                self.assertTrue(all(math.isfinite(x) for x in (energy,) + forces + virial))
                                self.assertGreater(max(abs(x) for x in forces), 1e-8)
                                connection.sendall(header('STATUS'))
                                self.assertEqual(receive(connection, 12), header('READY'))
                            connection.sendall(header(message))
                            result = process.wait(timeout=15)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                log = (work / 'output.log').read_text()
                if message == 'EXIT':
                    self.assertEqual(result, 0, log)
                    self.assertIn('Received EXIT from i-PI; closing normally.', log)
                    self.assertNotIn('MPI_ABORT', log)
                    self.assertNotIn('ERROR', log)
                else:
                    self.assertNotEqual(result, 0, log)
                    expected = ('Wrapper did not send positions' if stage == 'ready'
                                else 'Wrapper did not ask for forces')
                    self.assertIn(expected, log)

    def test_exit_waiting_for_positions(self):
        self.exchange('ready')

    def test_exit_waiting_for_getforce(self):
        self.exchange('have_data')

    def test_exit_after_force_reply(self):
        self.exchange('after_force')

    def test_invalid_header_waiting_for_positions(self):
        self.exchange('ready', 'INVALID')

    def test_invalid_header_waiting_for_getforce(self):
        self.exchange('have_data', 'INVALID')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmp', required=True)
    arguments, remaining = parser.parse_known_args()
    FixIPIExit.executable = str(Path(arguments.lmp).resolve())
    unittest.main(argv=[sys.argv[0]] + remaining)
