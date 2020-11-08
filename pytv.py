import time
import math
from pywebostv.connection import WebOSClient
from pywebostv.controls import InputControl, SystemControl, MediaControl


class PyTVCursor:
    '''
    Implements a smaller coordinate window of size self.width x self.height within the larger coordinate system.
    This window is shifted within the larger coordinate system such that the cursor drags the window along with it.

    The current x,y position of the window within the larger coodinate system is given by self.minx and self.miny

    The position of the cursor within the window is given by self._x and self._y

    The pos properties are used to pass the current external coordinate and retrieve the internal coordinate
    '''

    def __init__(self):
        # X, Y, Height, width of window in world coordinates +y up
        self.world_window_height = 100.0
        self.world_window_width = 120.0
        self.world_window_x = -60.0
        self.world_window_y = -50.0

        # Height and width of window in tv coordinates
        self.tv_window_half_height = 50
        self.tv_window_half_width = 85

        # Current cursor coordinate in tv coordinates, starts in center of screen +y down
        self.tv_cursor_x = 0
        self.tv_cursor_y = 0

        self.last_click_ms = 0
        self.paused = False

        self.input_control = None
        self.system_control = None
        self.media_control = None

        self.tv_context = {
            'client_key': 'c8bfd5128b861d24a0ffbf6644eb6f18',
            'local_address': '10.0.0.214'  # client.peer_address[0]
        }

        self.connect_and_register_to_tv()

    def update_world_coordinate(self, pos):

        if pos is None or len(pos) != 2:
            return

        # shift window in x direction
        self.world_window_x = min(pos[0], self.world_window_x)
        if pos[0] > self.world_window_x + self.world_window_width:
            self.world_window_x = pos[0] - self.world_window_width

        world_window_x_center = self.world_window_x + 0.5 * self.world_window_width
        x_unit_vector = (pos[0] - world_window_x_center) / (self.world_window_width * 0.5)

        # shift window in y direction and calculate vector
        self.world_window_y = min(pos[1], self.world_window_y)
        if pos[1] > self.world_window_y + self.world_window_height:
            self.world_window_y = pos[1] - self.world_window_height

        world_window_y_center = self.world_window_y + 0.5 * self.world_window_height
        y_unit_vector = (pos[1] - world_window_y_center) / (self.world_window_height * 0.5)

        new_x_coord = x_unit_vector * self.tv_window_half_width * -1
        new_y_coord = y_unit_vector * self.tv_window_half_height

        self.normalized_cursor_move(new_x_coord, new_y_coord)

    def normalized_cursor_move(self, new_x_coord, new_y_coord):
        try:
            while True:
                # always move in increments of 5, and move both x and y at once if possible
                x_step = 0 if int(new_x_coord) == self.tv_cursor_x else math.copysign(5, int(new_x_coord) - self.tv_cursor_x)
                y_step = 0 if int(new_y_coord) == self.tv_cursor_y else math.copysign(5, int(new_y_coord) - self.tv_cursor_y)
                if x_step == 0 and y_step == 0:
                    return
                self.input_control.move(x_step, y_step)
                self.tv_cursor_x += x_step / 5
                self.tv_cursor_y += y_step / 5

        except Exception as e:
            print('pytv move failed with {}'.format(e))
            self.input_control.connect_input()

    def connect_and_register_to_tv(self):
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
                self.system_control = SystemControl(self.client)
                self.media_control = MediaControl(self.client)

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

    def center(self):
        for i in range(18):
            tv.move(-20, -20)
            # time.sleep(0.01)
        for i in range(16):
            tv.move(15, 10)
            # time.sleep(0.01)

    def move(self, x, y):
        self.input_control.move(x, y)

    def toggle_pause(self):
        if self.paused:
            self.media_control.play()
            self.paused = False
        else:
            self.media_control.pause()
            self.paused = True

    def keypad_up(self):
        self.input_control.up()

    def keypad_down(self):
        self.input_control.down()

    def keypad_left(self):
        self.input_control.left()

    def keypad_right(self):
        self.input_control.right()

    def keypad_ok(self):
        self.input_control.ok()

    def keypad_back(self):
        self.input_control.back()

    def test_amazon_profiles(self):
        self.center()
        print('start')
        for i in range(9):
            self.move(-15, 0)
        time.sleep(1)
        self.center()
        for i in range(5):
            self.move(-15, 0)
        time.sleep(1)
        self.center()
        time.sleep(1)
        self.center()
        for i in range(6):
            self.move(15, 0)
        time.sleep(1)
        self.center()
        for i in range(8):
            self.move(15, 0)

    def test_multi_path(self):
        self.center()
        print('start')
        for i in range(8):
            self.move(-15, 0)
        self.system_control.notify("Path 1 complete")
        time.sleep(5)
        self.center()

        for i in range(10):
            self.move(-12, 0)
        self.system_control.notify("Path 2 complete")
        time.sleep(5)
        self.center()

        for i in range(12):
            self.move(-10, 0)
        self.system_control.notify("Path 3 complete")
        time.sleep(5)

    # Validates the TV width/height by moving cursor in a figure-eight pattern
    def test_figure_eight(self):
        # Right
        for i in range(85):
            tv.move(5, 0)
            time.sleep(.01)
        # Down
        for i in range(50):
            tv.move(0, 5)
            time.sleep(.01)
        # Left
        for i in range(85):
            tv.move(-5, 0)
            time.sleep(.01)
        # Up
        for i in range(50):
            tv.move(0, -5)
            time.sleep(.01)
        # Left
        for i in range(85):
            tv.move(-5, 0)
            time.sleep(.01)
        # Up
        for i in range(50):
            tv.move(0, -5)
            time.sleep(.01)
        # Right
        for i in range(85):
            tv.move(5, 0)
            time.sleep(.01)
        # Down
        for i in range(50):
            tv.move(0, 5)
            time.sleep(.01)

    def test_world_coordinate_translation(self):
        self.update_world_coordinate((0.0, 0.0))
        time.sleep(1)
        self.update_world_coordinate((50.0, 0.0))
        time.sleep(1)
        self.update_world_coordinate((50.0, 50.0))
        time.sleep(1)
        self.update_world_coordinate((0.0, 50.0))
        time.sleep(1)
        self.update_world_coordinate((0.0, 0.0))
        time.sleep(1)
        self.update_world_coordinate((-100.0, 0.0))
        time.sleep(1)
        self.update_world_coordinate((-200.0, 0.0))
        time.sleep(1)
        self.update_world_coordinate((0.0, 200.0))
        time.sleep(1)


# half-width=280
# half-height=175
# max step vert = 20
# max step horiz = 15

if __name__ == '__main__':
    tv = PyTVCursor()
    # tv.test_multi_path()
    #tv.test_figure_eight()
    tv.test_world_coordinate_translation()
    time.sleep(1)
