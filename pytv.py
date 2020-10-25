import time
from pywebostv.connection import WebOSClient
from pywebostv.controls import InputControl, SystemControl, MediaControl

tv_context = {
    'client_key': 'c8bfd5128b861d24a0ffbf6644eb6f18',
    'local_address': '10.0.0.214'  # client.peer_address[0]
}

#client = WebOSClient.discover()[0]
client = WebOSClient(tv_context['local_address'])

client.connect()
register_status = client.register(store=tv_context)

for status in register_status:
    if status == WebOSClient.PROMPTED:
        print("Please accept the connect on the TV!")
    elif status == WebOSClient.REGISTERED:
        print("Registration successful!")

#system = SystemControl(client)
#system.notify("This is a test")

#media = MediaControl(client)
#media.play()
#print("issued play")
#time.sleep(5)
#media.pause()
#print("issued pause")

input_control = InputControl(client)
input_control.connect_input()
input_control.move(20, 10)
time.sleep(1)
input_control.move(2, 2)
time.sleep(1)
input_control.move(4, 4)
input_control.click()

