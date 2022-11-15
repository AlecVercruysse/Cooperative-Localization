from ftplib import FTP
import zipfile
import os

if os.getcwd()[-5:] != "/data":
    # raise ValueError("call this script from the data directory!")
    try:
        os.chdir("data")
    except FileNotFoundError:
        raise FileNotFoundError("script couldn't find data/ dir")

ftp = FTP("asrl3.utias.utoronto.ca")
ftp.login()
ftp.cwd('MRCLAM')
# ftp.retrlines('LIST')
fnames = [f"MRCLAM{i}.zip" for i in range(1, 10)]

for fname in fnames:
    print(f"downloading and extracting {i} of 9")
    if os.path.exists(fname):
        print("already found, skipping download...")
    else:
        with open(fname, 'wb') as fp:
            ftp.retrbinary(f'RETR {fname}', fp.write)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(".")

ftp.quit()
