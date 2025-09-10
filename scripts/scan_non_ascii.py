p='e8_mind_server_M18.7.py'
with open(p, 'rb') as f:
    b=f.read()
try:
    b.decode('ascii')
    print('ASCII_ONLY')
except UnicodeDecodeError:
    s=b.decode('utf-8', errors='replace')
    for i, line in enumerate(s.splitlines(), 1):
        if any(ord(ch)>127 for ch in line):
            print(f"{i}: {line}")
