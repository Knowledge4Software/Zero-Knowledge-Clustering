"""Fresh pinned SSH transport per request; password stays in process memory."""
from pathlib import Path
import getpass
import json
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from tools.server_connection_20261101 import connect,execute,REMOTE


def main():
    password=getpass.getpass('SSH password: ')
    print('READY',flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('close'):break
        client=None
        try:
            client=connect(password);client.get_transport().set_keepalive(15)
            if 'upload' in request:
                with client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(45)
                    for item in request['upload']:
                        rel=Path(item['relative'])
                        assert not rel.is_absolute() and '..' not in rel.parts
                        sftp.put(str(REPO/rel),REMOTE+'/'+rel.as_posix())
                print(json.dumps(dict(uploaded=len(request['upload']))),flush=True)
            elif 'download' in request:
                with client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(45)
                    for item in request['download']:
                        rel=Path(item['remote_relative']);out=REPO/item['local_relative']
                        assert not rel.is_absolute() and '..' not in rel.parts
                        assert out.resolve().is_relative_to(REPO.resolve())
                        out.parent.mkdir(parents=True,exist_ok=True)
                        sftp.get(REMOTE+'/'+rel.as_posix(),str(out))
                print(json.dumps(dict(downloaded=len(request['download']))),flush=True)
            else:
                print('REMOTE_RESULT '+execute(client,REMOTE+'/.venv/bin/python -',
                      request['code'],timeout=request.get('timeout',55)),flush=True)
        except Exception as error:
            print(json.dumps(dict(request_error=str(error))),flush=True)
        finally:
            if client:client.close()


if __name__=='__main__':main()
