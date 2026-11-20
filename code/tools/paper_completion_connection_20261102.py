"""Session-local SSH controller; credentials live only in process memory."""
from pathlib import Path
import getpass
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.server_connection_20261101 import connect, execute, REMOTE


def main():
    client = connect(getpass.getpass('SSH password: '))
    print('CONNECTED', flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('close'):
                break
            try:
                if 'upload' in request:
                    with client.open_sftp() as sftp:
                        for item in request['upload']:
                            relative = Path(item['relative'])
                            if relative.is_absolute() or '..' in relative.parts:
                                raise ValueError('Relative workspace path required')
                            sftp.put(str(relative), REMOTE+'/'+relative.as_posix())
                    print(json.dumps({'uploaded': len(request['upload'])}), flush=True)
                else:
                    result = execute(client, REMOTE+'/.venv/bin/python -',
                                     request['code'], timeout=request.get('timeout', 55))
                    print('REMOTE_RESULT '+result, flush=True)
            except Exception as error:
                print(json.dumps({'request_error': str(error)}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
