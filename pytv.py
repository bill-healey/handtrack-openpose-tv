import time
from pywebostv.connection import WebOSClient
from pywebostv.controls import InputControl, SystemControl, MediaControl


class PyTvCursor:
    '''
    Implements a smaller coordinate window of size self.width x self.height within the larger coordinate system.
    This window is shifted within the larger coordinate system such that the cursor drags the window along with it.

    The current x,y position of the window within the larger coodinate system is given by self.minx and self.miny

    The position of the cursor within the window is given by self._x and self._y

    The pos properties are used to pass the current external coordinate and retrieve the internal coordinate
    '''

    # system = SystemControl(client)
    # system.notify("This is a test")

    # media = MediaControl(client)
    # media.play()
    # media.pause()

    def __init__(self):
        self.height = 100.0
        self.width = 100.0
        self._x = 50.0
        self._y = 50.0
        self.minx = 0.0
        self.miny = 0.0
        self.last_click_ms = 0
        self.input_control = None
        self.last_x = 0.0
        self.last_y = 0.0

        self.tv_context = {
            'client_key': 'c8bfd5128b861d24a0ffbf6644eb6f18',
            'local_address': '10.0.0.214'  # client.peer_address[0]
        }

        # self.client = WebOSClient.discover()[0]
        self.client = WebOSClient(self.tv_context['local_address'])

        self.client.connect()
        register_status = self.client.register(store=self.tv_context)

        for status in register_status:
            if status == WebOSClient.PROMPTED:
                print("Please accept the connect on the TV!")
            elif status == WebOSClient.REGISTERED:
                print("Registration successful!")
                self.input_control = InputControl(self.client)
                self.input_control.connect_input()

    @property
    def pos(self):
        return self._x, self._y

    @pos.setter
    def pos(self, pos):

        if pos is None or len(pos) != 2:
            return

        if self.last_x is None or self.last_y is None:
            self.last_x = pos[0]
            self.last_y = pos[1]
            return

        # # shift window in x direction
        # self.minx = min(pos[0], self.minx)
        # if pos[0] > self.minx + self.width:
        #     self.minx = pos[0] - self.width
        # self._x = self.width - (pos[0] - self.minx)
        #
        # # shift window in y direction
        # self.miny = min(pos[1], self.miny)
        # if pos[1] > self.miny + self.height:
        #     self.miny = pos[1] - self.height
        # self._y = pos[1] - self.miny

        # move the cursor
        try:
            move_x = (pos[0]-self.last_x) * -2.0
            move_y = (pos[1]-self.last_y) * 2.0
            print('move {},{}'.format(move_x, move_y))
            self.input_control.move(move_x, move_y)
            self.last_x = pos[0]
            self.last_y = pos[1]
            time.sleep(.1)
        except Exception as e:
            print('pytv move failed with {}'.format(e))
            self.input_control.connect_input()

    def click(self):
        # De-bounce the click
        click_time = int(round(time.time() * 1000))
        if click_time < self.last_click_ms + 500:
            return
        self.last_click_ms = click_time

        # perform the click
        try:
            print('click emulated')
            self.input_control.click()
        except Exception as e:
            print('pytv click failed with {}'.format(e))
            self.input_control.connect_input()
